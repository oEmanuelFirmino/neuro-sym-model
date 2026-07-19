"""Experimento no dataset de Hinton (1986) — segundo domínio real (M2).

Protocolo (evidência completa gerada automaticamente em
`experiments/evidence/hinton_family/`):
1. Treina os 8 predicados base + 4 baselines planos (embeddings
   compartilhados), com split treino/validação nos fatos base — a curva de
   validação por época alimenta os gráficos de treino (save_run_report).
2. Consistência dedutiva: os 4 predicados derivados (uncle/aunt/nephew/niece),
   NUNCA treinados, são avaliados pela fórmula de prova composta — verdade
   média em pares derivados verdadeiros vs. falsos.
3. Explicabilidade arquitetural: massa de gradiente no intermediário z do
   caminho de prova + deleção causal, composta vs. plana.

Uso: `uv run python experiments/hinton_family/run_family.py`
"""

import json
import logging
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.neurosym.tensor.backend import set_backend

set_backend("numpy")
logging.disable(logging.CRITICAL)

import numpy as np

from experiments.hinton_family.domain import (
    BASE_RELATIONS,
    DERIVED_RELATIONS,
    build_family_grounding_env,
    derived_proof_formula,
    generate_family,
)
from experiments.reporting import EVIDENCE_ROOT, save_run_report
from src.neurosym.explainability.metrics import compute_influences, deletion_curve
from src.neurosym.interpreter import Interpreter
from src.neurosym.logic import Atom, Constant
from src.neurosym.module.module import Linear, ReLU, Sequential, Sigmoid
from src.neurosym.training.callbacks import Callback
from src.neurosym.training.optimizer import AdamW
from src.neurosym.training.trainer import Trainer

SEED = 0
EMBEDDING_DIM = 8
HIDDEN = 16
EPOCHS = 500
VAL_FRAC = 0.15

CONFIG = dict(
    dataset="hinton_family_english_tree_1986",
    seed=SEED,
    embedding_dim=EMBEDDING_DIM,
    hidden=HIDDEN,
    epochs=EPOCHS,
    val_frac=VAL_FRAC,
)


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


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    data = generate_family(seed=SEED)
    grounding_env = build_family_grounding_env(data.entities, EMBEDDING_DIM, SEED)
    predicate_map = {r: _predicate() for r in BASE_RELATIONS}
    predicate_map.update({f"{r}_flat": _predicate() for r in DERIVED_RELATIONS})
    interpreter = Interpreter(predicate_map, grounding_env)

    n_val = int(len(data.base_facts) * VAL_FRAC)
    val_facts = data.base_facts[:n_val]
    train_facts = data.base_facts[n_val:] + data.flat_derived_facts

    all_params = list(grounding_env.values())
    for model in predicate_map.values():
        all_params.extend(model.parameters())

    capture = _Capture()
    trainer = Trainer(
        interpreter,
        AdamW(all_params, lr=1e-2, weight_decay=1e-3),
        epochs=EPOCHS,
        callbacks=[capture],
    )

    print(f"=== treinando 12 predicados ({EPOCHS} épocas, {len(train_facts)} fatos) ===", flush=True)
    t0 = time.time()
    trainer.fit(rules=[], facts=train_facts, val_facts=val_facts)
    train_seconds = time.time() - t0
    val_acc = capture.epoch_logs[-1].get("val_accuracy")
    print(f"  treino: {train_seconds:.0f}s | val acc (fatos base held-out): {val_acc:.3f}", flush=True)

    # --- relatório de treino padrão: curvas + tabela, sempre gerados ---
    train_result = {
        "seed": SEED,
        "t_g": None,
        "final_val_accuracy": val_acc,
        "test_accuracy": None,
        "final_l1_penalty": capture.epoch_logs[-1].get("l1_penalty"),
        "epoch_logs": capture.epoch_logs,
    }
    evidence_dir = save_run_report(
        "hinton_family", train_result, config=CONFIG, timestamp=False
    )

    # --- consistência dedutiva por relação derivada ---
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
        random.Random(SEED).shuffle(negatives)
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
            "composed_positives": _agg(pos_truth),
            "composed_negatives": _agg(neg_truth),
            "flat_positives": _agg(flat_pos),
            "flat_negatives": _agg(flat_neg),
        }
        print(
            f"  {relation}: composta {consistency[relation]['composed_positives']['mean']:.3f}/"
            f"{consistency[relation]['composed_negatives']['mean']:.3f} | "
            f"plana {consistency[relation]['flat_positives']['mean']:.3f}/"
            f"{consistency[relation]['flat_negatives']['mean']:.3f} (pos/neg)",
            flush=True,
        )

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
                    "query": f"{relation}({x},{y})",
                    "intermediate": z,
                    "intermediate_mass_composed": inf_c[z] / total_c if total_c else None,
                    "intermediate_mass_flat": inf_f[z] / total_f if total_f else None,
                    "deletion_delta_composed": del_c[0] - del_c[-1],
                    "deletion_delta_flat": del_f[0] - del_f[-1],
                }
            )

    explainability = {
        "n_queries": len(per_query),
        "intermediate_mass_composed": _agg(
            [q["intermediate_mass_composed"] for q in per_query]
        ),
        "intermediate_mass_flat": _agg([q["intermediate_mass_flat"] for q in per_query]),
        "deletion_delta_composed": _agg([q["deletion_delta_composed"] for q in per_query]),
        "deletion_delta_flat": _agg([q["deletion_delta_flat"] for q in per_query]),
        "per_query": per_query,
    }
    print(
        "  explicabilidade (massa intermediário): composta "
        f"{explainability['intermediate_mass_composed']['mean']:.3f} vs plana "
        f"{explainability['intermediate_mass_flat']['mean']:.3f} | "
        f"Δ deleção: {explainability['deletion_delta_composed']['mean']:.3f} vs "
        f"{explainability['deletion_delta_flat']['mean']:.3f}",
        flush=True,
    )

    # --- evidência dedicada (consistência + explicabilidade) ---
    with open(evidence_dir / "derived_report.json", "w", encoding="utf-8") as f:
        json.dump(
            {"config": CONFIG, "consistency": consistency, "explainability": explainability},
            f,
            indent=2,
        )

    lines = [
        "# Hinton family (1986) — deduced relations via proof DAG",
        "",
        f"Base-fact held-out accuracy: {val_acc:.4f} | training: {train_seconds:.0f}s",
        "",
        "## Deductive consistency (never trained on derived relations)",
        "",
        "| Relation | Composed pos/neg | Flat (trained) pos/neg |",
        "|---|---|---|",
    ]
    for relation, entry in consistency.items():
        lines.append(
            f"| {relation} | {entry['composed_positives']['mean']:.3f} / "
            f"{entry['composed_negatives']['mean']:.3f} | "
            f"{entry['flat_positives']['mean']:.3f} / "
            f"{entry['flat_negatives']['mean']:.3f} |"
        )
    lines += [
        "",
        "## Architectural explainability",
        "",
        "| Metric | Composed | Flat |",
        "|---|---|---|",
        f"| Gradient mass on path intermediate | {explainability['intermediate_mass_composed']['mean']:.4f} | {explainability['intermediate_mass_flat']['mean']:.4f} |",
        f"| Δ truth after deleting intermediate | {explainability['deletion_delta_composed']['mean']:.4f} | {explainability['deletion_delta_flat']['mean']:.4f} |",
        "",
    ]
    (evidence_dir / "derived_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("evidence:", evidence_dir, flush=True)


if __name__ == "__main__":
    main()
