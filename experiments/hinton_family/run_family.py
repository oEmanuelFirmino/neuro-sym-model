"""Experimento no dataset de Hinton (1986) — segundo domínio real (M2).

Protocolo (evidência completa gerada automaticamente em
`experiments/evidence/hinton_family/`):
1. Treina os 8 predicados base + 4 baselines planos (embeddings
   compartilhados) em TODOS os fatos base. As relações derivadas permanecem
   nunca treinadas (alvo de generalização); reporta-se a acurácia de ajuste
   dos predicados base como diagnóstico.
2. Consistência dedutiva: os 4 predicados derivados (uncle/aunt/nephew/niece),
   NUNCA treinados, são avaliados pela fórmula de prova composta — verdade
   média em pares derivados verdadeiros vs. falsos.
3. Explicabilidade arquitetural: massa de gradiente no intermediário z do
   caminho de prova + deleção causal, composta vs. plana.

Multi-semente (M-1 do parecer): por padrão roda 5 sementes independentes e agrega
os resultados por **média ± desvio-padrão entre sementes** (variância de
inicialização), não apenas entre consultas de um único modelo. Cada semente
reconstrói embeddings, predicados e o split treino/validação do zero.

Uso:
    uv run python experiments/hinton_family/run_family.py                # 5 sementes
    uv run python experiments/hinton_family/run_family.py --seeds 0,1,2  # subconjunto
    uv run python experiments/hinton_family/run_family.py --seeds 0 --epochs 20  # smoke
"""

import argparse
import json
import logging
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
# console Windows usa cp1252; sem isto, prints com caracteres como Δ quebram o run
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.neurosym.tensor.backend import set_backend

set_backend("numpy")
logging.disable(logging.CRITICAL)

import numpy as np
from joblib import Parallel, delayed

from experiments.hinton_family.domain import (
    BASE_RELATIONS,
    DERIVED_RELATIONS,
    build_family_grounding_env,
    derived_proof_formula,
    generate_family,
)
from experiments.reporting import EVIDENCE_ROOT
from src.neurosym.explainability.metrics import compute_influences, deletion_curve
from src.neurosym.interpreter import Interpreter
from src.neurosym.logic import Atom, Constant
from src.neurosym.module.module import Linear, ReLU, Sequential, Sigmoid
from src.neurosym.training.callbacks import Callback
from src.neurosym.training.optimizer import AdamW
from src.neurosym.training.trainer import Trainer

DEFAULT_SEEDS = [0, 1, 2, 3, 4]
EMBEDDING_DIM = 8
HIDDEN = 16
EPOCHS = 500


class _Capture(Callback):
    def __init__(self):
        super().__init__()
        self.epoch_logs = []

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        self.epoch_logs.append(dict(logs or {}))


def _predicate():
    return Sequential(
        Linear(2 * EMBEDDING_DIM, HIDDEN), ReLU(), Linear(HIDDEN, 1), Sigmoid()
    )


def _truth(interpreter, formula) -> float:
    result = interpreter.eval_formula(formula, {})
    return result._flatten(result.data)[0]


def _agg(values):
    if not values:
        return {"mean": None, "std": None, "n": 0}
    return {
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def _mean(values):
    """Média escalar sobre uma lista (média entre consultas de UMA semente)."""
    vals = [v for v in values if v is not None]
    return statistics.fmean(vals) if vals else None


def run_single(seed: int, epochs: int) -> dict:
    """Treina os 12 predicados e avalia consistência + explicabilidade para uma
    única semente, retornando os escalares (média entre consultas) daquela
    semente. Cada semente reconstrói embeddings, predicados e split do zero.

    Sementes são independentes entre si, então `main` roda uma por processo em
    paralelo (joblib/loky). Como o script roda como `__main__`, loky serializa
    `run_single` via cloudpickle (função definida em `__main__`) em vez de por
    referência de módulo -- o código de nível de módulo deste arquivo (que
    fixa a codificação do stdout, o backend de tensor e o nível de log) NÃO
    reexecuta no worker, então os três são refeitos aqui dentro.
    """
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    set_backend("numpy")
    logging.disable(logging.CRITICAL)
    random.seed(seed)
    np.random.seed(seed)

    data = generate_family(seed=seed)
    grounding_env = build_family_grounding_env(data.entities, EMBEDDING_DIM, seed)
    predicate_map = {r: _predicate() for r in BASE_RELATIONS}
    predicate_map.update({f"{r}_flat": _predicate() for r in DERIVED_RELATIONS})
    interpreter = Interpreter(predicate_map, grounding_env)

    # M-1: os 8 predicados base são treinados em TODOS os seus fatos. As relações
    # derivadas (uncle/aunt/nephew/niece) permanecem NUNCA treinadas — são o alvo
    # de generalização. A versão anterior segurava 15% dos fatos base como
    # validação; como o split é embaralhado por semente, elos positivos das cadeias
    # de prova caíam no holdout em algumas sementes, starvando a composição e
    # colapsando-a a ~0 de forma dependente de inicialização (artefato metodológico
    # confirmado: ver diagnóstico da correção M-1). Fatos base têm cobertura
    # completa, então segurá-los não media generalização útil para esta tarefa.
    train_facts = data.base_facts + data.flat_derived_facts

    all_params = list(grounding_env.values())
    for model in predicate_map.values():
        all_params.extend(model.parameters())

    capture = _Capture()
    trainer = Trainer(
        interpreter,
        AdamW(all_params, lr=1e-2, weight_decay=1e-3),
        epochs=epochs,
        callbacks=[capture],
    )

    print(f"  [seed {seed}] treinando 12 predicados ({epochs} épocas, {len(train_facts)} fatos) ...", flush=True)
    t0 = time.time()
    trainer.fit(rules=[], facts=train_facts)
    train_seconds = time.time() - t0
    # diagnóstico de ajuste: acurácia dos predicados base sobre todos os fatos base
    base_fit_acc = trainer.evaluate_accuracy(data.base_facts)

    # --- consistência dedutiva por relação derivada (escalar por relação) ---
    consistency = {}
    for relation in DERIVED_RELATIONS:
        gold = data.derived_gold[relation]
        positives = {(x, y) for x, y, _ in gold}
        negatives = [
            (x, y)
            for x in data.entities
            for y in data.entities
            if x != y and (x, y) not in positives
        ]
        random.Random(seed).shuffle(negatives)
        negatives = negatives[: len(positives) * 5]

        pos_truth = [
            _truth(interpreter, derived_proof_formula(relation, x, y, data.entities))
            for x, y in positives
        ]
        neg_truth = [
            _truth(interpreter, derived_proof_formula(relation, x, y, data.entities))
            for x, y in negatives
        ]
        flat_pos = [
            _truth(interpreter, Atom(f"{relation}_flat", [Constant(x), Constant(y)]))
            for x, y in positives
        ]
        flat_neg = [
            _truth(interpreter, Atom(f"{relation}_flat", [Constant(x), Constant(y)]))
            for x, y in negatives
        ]
        consistency[relation] = {
            "composed_pos": _mean(pos_truth),
            "composed_neg": _mean(neg_truth),
            "flat_pos": _mean(flat_pos),
            "flat_neg": _mean(flat_neg),
        }

    # --- explicabilidade: massa no intermediário + deleção causal ---
    per_query = []
    for relation in DERIVED_RELATIONS:
        for x, y, z in sorted(data.derived_gold[relation]):
            composed = derived_proof_formula(relation, x, y, data.entities)
            flat = Atom(f"{relation}_flat", [Constant(x), Constant(y)])

            inf_c = compute_influences(interpreter, composed)
            inf_f = compute_influences(interpreter, flat)
            total_c, total_f = sum(inf_c.values()), sum(inf_f.values())

            del_c = deletion_curve(interpreter, composed, [z])
            del_f = deletion_curve(interpreter, flat, [z])

            per_query.append(
                {
                    "intermediate_mass_composed": inf_c[z] / total_c if total_c else None,
                    "intermediate_mass_flat": inf_f[z] / total_f if total_f else None,
                    "deletion_delta_composed": del_c[0] - del_c[-1],
                    "deletion_delta_flat": del_f[0] - del_f[-1],
                }
            )

    explainability = {
        "intermediate_mass_composed": _mean([q["intermediate_mass_composed"] for q in per_query]),
        "intermediate_mass_flat": _mean([q["intermediate_mass_flat"] for q in per_query]),
        "deletion_delta_composed": _mean([q["deletion_delta_composed"] for q in per_query]),
        "deletion_delta_flat": _mean([q["deletion_delta_flat"] for q in per_query]),
    }
    print(
        f"  [seed {seed}] {train_seconds:.0f}s | base-fit acc {base_fit_acc:.3f} | "
        f"massa intermed {explainability['intermediate_mass_composed']:.3f} | "
        f"Δdel {explainability['deletion_delta_composed']:.3f}",
        flush=True,
    )
    return {
        "seed": seed,
        "train_seconds": train_seconds,
        "base_fit_accuracy": base_fit_acc,
        "consistency": consistency,
        "explainability": explainability,
        "n_queries": len(per_query),
    }


def main():
    parser = argparse.ArgumentParser(description="Hinton family multi-semente (M-1)")
    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(map(str, DEFAULT_SEEDS)),
        help="Lista de sementes separadas por vírgula (padrão: 0,1,2,3,4).",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]

    print(
        f"=== Hinton family multi-semente: {len(seeds)} sementes {seeds}, "
        f"{args.epochs} épocas cada ===",
        flush=True,
    )
    t_start = time.time()
    runs = Parallel(n_jobs=-1, backend="loky")(
        delayed(run_single)(seed, args.epochs) for seed in seeds
    )

    # --- agregação ENTRE sementes (variância de inicialização) ---
    consistency_agg = {
        relation: {
            field: _agg([r["consistency"][relation][field] for r in runs])
            for field in ("composed_pos", "composed_neg", "flat_pos", "flat_neg")
        }
        for relation in DERIVED_RELATIONS
    }
    explainability_agg = {
        field: _agg([r["explainability"][field] for r in runs])
        for field in (
            "intermediate_mass_composed",
            "intermediate_mass_flat",
            "deletion_delta_composed",
            "deletion_delta_flat",
        )
    }
    base_fit_agg = _agg([r["base_fit_accuracy"] for r in runs])

    config = dict(
        dataset="hinton_family_english_tree_1986",
        seeds=seeds,
        embedding_dim=EMBEDDING_DIM,
        hidden=HIDDEN,
        epochs=args.epochs,
        base_training="todos os fatos base (sem holdout); derivadas nunca treinadas",
        aggregation="mean ± pop-std ENTRE sementes (cada valor por semente é a média entre consultas)",
    )
    payload = {
        "config": config,
        "base_fact_fit_accuracy": base_fit_agg,
        "consistency": consistency_agg,
        "explainability": explainability_agg,
        "runs": runs,
    }

    out_dir = EVIDENCE_ROOT / "hinton_family_multiseed"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "derived_report.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    def _cell(agg):
        return f"{agg['mean']:.3f} ± {agg['std']:.3f}" if agg["mean"] is not None else "—"

    lines = [
        "# Hinton family (1986) — deduced relations via proof DAG (multi-seed)",
        "",
        f"Seeds: {seeds} (n={len(seeds)}) · {args.epochs} épocas · agregação mean ± std ENTRE sementes.",
        f"Base-fact fit accuracy (todos os fatos base treinados): {_cell(base_fit_agg)}",
        "",
        "## Deductive consistency (never trained on derived relations)",
        "",
        "| Relation | Composed pos | Composed neg | Flat (trained) pos | Flat neg |",
        "|---|---|---|---|---|",
    ]
    for relation in DERIVED_RELATIONS:
        e = consistency_agg[relation]
        lines.append(
            f"| {relation} | {_cell(e['composed_pos'])} | {_cell(e['composed_neg'])} | "
            f"{_cell(e['flat_pos'])} | {_cell(e['flat_neg'])} |"
        )
    lines += [
        "",
        "## Architectural explainability",
        "",
        "| Metric | Composed | Flat |",
        "|---|---|---|",
        f"| Gradient mass on path intermediate | {_cell(explainability_agg['intermediate_mass_composed'])} | {_cell(explainability_agg['intermediate_mass_flat'])} |",
        f"| Δ truth after deleting intermediate | {_cell(explainability_agg['deletion_delta_composed'])} | {_cell(explainability_agg['deletion_delta_flat'])} |",
        "",
        "Dispersion is across independent seeds (initialization variance), not "
        "across queries of a single model.",
        "",
    ]
    (out_dir / "derived_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n=== concluído em {time.time() - t_start:.0f}s ===", flush=True)
    print("evidence:", out_dir, flush=True)
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
