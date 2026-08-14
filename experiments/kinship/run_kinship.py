"""Experimento central do artigo reposicionado: explicabilidade e consistência
lógica na inferência composta via DAG de prova (domínio de parentesco).

Protocolo:
1. Treina os predicados `parent` e `ancestor_flat` (embeddings compartilhados)
   sobre os fatos do domínio. `ancestor` composto NUNCA é treinado — é derivado
   em tempo de consulta pela fórmula de prova (Product T-norm sobre o DAG).
2. **Consistência lógica / transferência de raciocínio**: mede se a inferência
   composta discrimina pares ancestral verdadeiros de falsos usando apenas o
   predicado base aprendido — raciocínio dedutivo emergindo da composição, sem
   supervisão direta do conceito derivado.
3. **Explicabilidade arquitetural**: sobre as consultas com encadeamento
   (caminho >= 2), compara a inferência composta com o predicado plano em:
   - fração da massa de gradiente nos intermediários do caminho (a "assinatura"
     do raciocínio — estruturalmente zero no plano);
   - efeito de deletar os intermediários sobre o grau de verdade previsto
     (dependência causal do caminho de prova);
   - concentração no fecho (constantes da consulta + intermediários).

Evidência salva via experiments/reporting + JSON dedicado.

Multi-semente (M-1 do parecer): por padrão roda 5 sementes independentes e agrega
os resultados por **média ± desvio-padrão entre sementes** (variância de
inicialização), não apenas entre consultas de um único modelo. Cada semente
reconstrói embeddings, predicados e amostragem de negativos do zero.

Uso:
    uv run python experiments/kinship/run_kinship.py                 # 5 sementes
    uv run python experiments/kinship/run_kinship.py --seeds 0,1,2   # subconjunto
    uv run python experiments/kinship/run_kinship.py --seeds 0 --epochs 20  # smoke
"""

import argparse
import json
import logging
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

import random

from experiments.kinship.domain import (
    ancestor_proof_formula,
    build_kinship_grounding_env,
    generate_kinship,
)
from experiments.reporting import EVIDENCE_ROOT
from src.neurosym.explainability.metrics import (
    compute_influences,
    concentration,
    deletion_curve,
)
from src.neurosym.interpreter import Interpreter
from src.neurosym.logic import Atom, Constant
from src.neurosym.module.module import Linear, ReLU, Sequential, Sigmoid
from src.neurosym.training.optimizer import AdamW
from src.neurosym.training.trainer import Trainer

DEFAULT_SEEDS = [0, 1, 2, 3, 4]
EMBEDDING_DIM = 8
HIDDEN = 16
EPOCHS = 500
PROOF_DEPTH = 3  # cobre os caminhos mais longos do domínio (2 intermediários)


def _predicate(in_features: int) -> Sequential:
    return Sequential(
        Linear(in_features, HIDDEN), ReLU(), Linear(HIDDEN, 1), Sigmoid()
    )


def _truth(interpreter: Interpreter, formula) -> float:
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
    """Treina e avalia o domínio para uma única semente, retornando as métricas
    escalares (média entre consultas) daquela semente. Cada semente reconstrói
    embeddings, predicados e amostragem de negativos do zero."""
    random.seed(seed)
    import numpy as np

    np.random.seed(seed)

    # negatives_per_positive alto o bastante para cobrir TODOS os não-arcos:
    # com Product t-conorm, o OR da fórmula de prova agrega ~|E| ramos; ramos
    # de pares nunca vistos no treino pontuam ~0.5 e saturam a disjunção
    # (1 - prod(1-x_i) -> 1). A composição dedutiva exige o predicado base
    # calibrado em todo o domínio — limitação metodológica a registrar no texto.
    data = generate_kinship(seed=seed, negatives_per_positive=20)
    grounding_env = build_kinship_grounding_env(data.entities, EMBEDDING_DIM, seed)
    predicate_map = {
        "parent": _predicate(2 * EMBEDDING_DIM),
        "ancestor_flat": _predicate(2 * EMBEDDING_DIM),
    }
    interpreter = Interpreter(predicate_map, grounding_env)

    all_params = list(grounding_env.values())
    for model in predicate_map.values():
        all_params.extend(model.parameters())

    trainer = Trainer(
        interpreter,
        AdamW(all_params, lr=1e-2, weight_decay=1e-3),
        epochs=epochs,
    )

    print(f"  [seed {seed}] treinando parent + ancestor_flat ({epochs} épocas) ...", flush=True)
    t0 = time.time()
    trainer.fit(rules=[], facts=data.parent_facts + data.ancestor_flat_facts)
    train_seconds = time.time() - t0

    parent_train_acc = trainer.evaluate_accuracy(data.parent_facts)
    flat_train_acc = trainer.evaluate_accuracy(data.ancestor_flat_facts)

    # --- 2. consistência lógica: a composição discrimina sem treino direto? ---
    positive_pairs = [(x, z) for x, z, _ in data.chained_queries]
    ancestor_set = set(positive_pairs) | set(data.edges)
    negative_pairs = [
        (x, z)
        for x in data.entities
        for z in data.entities
        if x != z and (x, z) not in ancestor_set
    ][: len(positive_pairs) * 2]

    composed_pos = [
        _truth(interpreter, ancestor_proof_formula(x, z, data.entities, PROOF_DEPTH))
        for x, z in positive_pairs
    ]
    composed_neg = [
        _truth(interpreter, ancestor_proof_formula(x, z, data.entities, PROOF_DEPTH))
        for x, z in negative_pairs
    ]
    flat_pos = [
        _truth(interpreter, Atom("ancestor_flat", [Constant(x), Constant(z)]))
        for x, z in positive_pairs
    ]
    flat_neg = [
        _truth(interpreter, Atom("ancestor_flat", [Constant(x), Constant(z)]))
        for x, z in negative_pairs
    ]

    # --- 3. explicabilidade arquitetural sobre consultas encadeadas ---
    per_query = []
    for x, z, intermediates in data.chained_queries:
        composed = ancestor_proof_formula(x, z, data.entities, PROOF_DEPTH)
        flat = Atom("ancestor_flat", [Constant(x), Constant(z)])
        closure = {x, z} | intermediates

        inf_composed = compute_influences(interpreter, composed)
        inf_flat = compute_influences(interpreter, flat)

        total_c = sum(inf_composed.values())
        total_f = sum(inf_flat.values())
        mid_mass_composed = (
            sum(inf_composed[m] for m in intermediates) / total_c if total_c else None
        )
        mid_mass_flat = (
            sum(inf_flat[m] for m in intermediates) / total_f if total_f else None
        )

        order = sorted(intermediates)  # dependência causal: deletar SÓ os intermediários
        del_composed = deletion_curve(interpreter, composed, order)
        del_flat = deletion_curve(interpreter, flat, order)

        per_query.append(
            {
                "intermediate_mass_composed": mid_mass_composed,
                "intermediate_mass_flat": mid_mass_flat,
                "concentration_composed": concentration(inf_composed, closure),
                "concentration_flat": concentration(inf_flat, closure),
                "deletion_delta_composed": del_composed[0] - del_composed[-1],
                "deletion_delta_flat": del_flat[0] - del_flat[-1],
            }
        )

    scalars = {
        "composed_pos": _mean(composed_pos),
        "composed_neg": _mean(composed_neg),
        "flat_pos": _mean(flat_pos),
        "flat_neg": _mean(flat_neg),
        "intermediate_mass_composed": _mean([q["intermediate_mass_composed"] for q in per_query]),
        "intermediate_mass_flat": _mean([q["intermediate_mass_flat"] for q in per_query]),
        "concentration_composed": _mean([q["concentration_composed"] for q in per_query]),
        "concentration_flat": _mean([q["concentration_flat"] for q in per_query]),
        "deletion_delta_composed": _mean([q["deletion_delta_composed"] for q in per_query]),
        "deletion_delta_flat": _mean([q["deletion_delta_flat"] for q in per_query]),
    }
    print(
        f"  [seed {seed}] {train_seconds:.0f}s | composta pos/neg "
        f"{scalars['composed_pos']:.3f}/{scalars['composed_neg']:.3f} | "
        f"massa intermed {scalars['intermediate_mass_composed']:.3f} | "
        f"Δdel {scalars['deletion_delta_composed']:.3f}",
        flush=True,
    )
    return {
        "seed": seed,
        "train_seconds": train_seconds,
        "train_accuracy": {"parent": parent_train_acc, "ancestor_flat": flat_train_acc},
        "scalars": scalars,
        "n_positive_pairs": len(positive_pairs),
        "n_negative_pairs": len(negative_pairs),
        "n_chained_queries": len(per_query),
    }


def main():
    parser = argparse.ArgumentParser(description="Kinship multi-semente (M-1)")
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
        f"=== Kinship multi-semente: {len(seeds)} sementes {seeds}, "
        f"{args.epochs} épocas cada ===",
        flush=True,
    )
    t_start = time.time()
    runs = [run_single(seed, args.epochs) for seed in seeds]

    # --- agregação ENTRE sementes (variância de inicialização) ---
    scalar_keys = list(runs[0]["scalars"].keys())
    aggregate = {
        key: _agg([r["scalars"][key] for r in runs]) for key in scalar_keys
    }
    payload = {
        "config": {
            "seeds": seeds,
            "embedding_dim": EMBEDDING_DIM,
            "hidden": HIDDEN,
            "epochs": args.epochs,
            "proof_depth": PROOF_DEPTH,
            "negatives_per_positive": 20,
            "aggregation": "mean ± pop-std ENTRE sementes (cada valor por semente é a média entre consultas)",
        },
        "aggregate": aggregate,
        "runs": [
            {k: v for k, v in r.items() if k != "epoch_logs"} for r in runs
        ],
    }

    out_dir = EVIDENCE_ROOT / "kinship_proof_dag_multiseed"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "kinship_report.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    def _cell(key):
        a = aggregate[key]
        return f"{a['mean']:.4f} ± {a['std']:.4f}" if a["mean"] is not None else "—"

    lines = [
        "# Kinship — proof-DAG composed inference vs flat predicate (multi-seed)",
        "",
        f"Seeds: {seeds} (n={len(seeds)}) · {args.epochs} épocas · agregação mean ± std ENTRE sementes.",
        "",
        "| Metric | Composed (proof DAG) | Flat predicate |",
        "|---|---|---|",
        f"| Truth on true ancestor pairs | {_cell('composed_pos')} | {_cell('flat_pos')} |",
        f"| Truth on false ancestor pairs | {_cell('composed_neg')} | {_cell('flat_neg')} |",
        f"| Gradient mass on path intermediates | {_cell('intermediate_mass_composed')} | {_cell('intermediate_mass_flat')} |",
        f"| Concentration on closure | {_cell('concentration_composed')} | {_cell('concentration_flat')} |",
        f"| Δ truth after deleting intermediates | {_cell('deletion_delta_composed')} | {_cell('deletion_delta_flat')} |",
        "",
        "Composed inference is never trained on ancestor pairs: discrimination "
        "and path-dependence emerge from composing the learned parent predicate "
        "through the proof DAG (Product T-norms). Dispersion is now across "
        f"independent seeds (initialization variance), not across queries of a single model.",
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n=== concluído em {time.time() - t_start:.0f}s ===", flush=True)
    print("evidence:", out_dir, flush=True)
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
