"""Avaliação quantitativa de explicabilidade (M5) no domínio de Adição Modular.

Treina um DLG em p=13 até o estado memorizado (o modelo atual não generaliza —
ver plano; a explicabilidade é medida sobre o que o modelo de fato aprendeu) e
roda os protocolos de fidelidade de `explainability/metrics.py` sobre N
consultas de validação:

  - concentração do gradiente nas constantes da tripla consultada;
  - deletion/insertion AUC na ordem do gradiente vs. ordem aleatória.

Evidência salva via `experiments/reporting.py` + um JSON dedicado com o
relatório de explicabilidade.

Uso: `uv run python experiments/modular_addition/run_explainability.py`
"""

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.neurosym.tensor.backend import set_backend

set_backend("numpy")
logging.disable(logging.CRITICAL)

from experiments.modular_addition.run import make_dlg_build_fn
from experiments.reporting import EVIDENCE_ROOT, save_run_report
from experiments.run_multiseed import run_single_seed
from src.neurosym.explainability.metrics import evaluate_queries

P = 13
EPOCHS = 400  # suficiente para memorizar (l_data ~0 na época ~300)
SEED = 0
N_QUERIES = 25

CONFIG = dict(
    p=P,
    embedding_dim=8,
    hidden=24,
    epochs=EPOCHS,
    val_eval_every=20,
    gamma_l1=1e-4,
)


def main():
    print(f"=== treinando DLG p={P} por {EPOCHS} épocas (seed {SEED}) ===", flush=True)
    build_fn = make_dlg_build_fn(**CONFIG)

    # build/treino manual (não via run_single_seed) para manter o interpreter
    # treinado em mãos para a avaliação de explicabilidade.
    import random as _random

    _random.seed(SEED)
    import numpy as _np

    _np.random.seed(SEED)
    spec = build_fn(SEED)

    t0 = time.time()
    spec.trainer.fit(rules=spec.rules, facts=spec.facts, val_facts=spec.val_facts)
    print(f"  treino: {time.time() - t0:.1f}s", flush=True)

    # consultas: fatos positivos de validação; relevantes = constantes da tripla
    positive_val = [(f, t) for f, t in spec.val_facts if t >= 0.5][:N_QUERIES]
    queries = [
        (formula, {term.name for term in formula.terms})
        for formula, _ in positive_val
    ]

    print(f"=== avaliando explicabilidade em {len(queries)} consultas ===", flush=True)
    t0 = time.time()
    report = evaluate_queries(spec.interpreter, queries, seed=SEED)
    elapsed = time.time() - t0

    agg = {k: report[k] for k in report if k != "per_query"}
    print(json.dumps(agg, indent=2), flush=True)
    print(f"  avaliação: {elapsed:.1f}s", flush=True)

    out_dir = EVIDENCE_ROOT / "explainability_dlg_p13"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "explainability_report.json", "w", encoding="utf-8") as f:
        json.dump({"config": CONFIG, "report": report}, f, indent=2)

    # tabela resumo em markdown
    lines = [
        "# Explainability — DLG p=13 (memorized state)",
        "",
        f"Config: `{CONFIG}`",
        "",
        "| Metric | Mean ± std (n) |",
        "|---|---|",
    ]
    for key in (
        "concentration",
        "deletion_auc_gradient",
        "deletion_auc_random",
        "insertion_auc_gradient",
        "insertion_auc_random",
    ):
        stats = report[key]
        value = (
            f"{stats['mean']:.4f} ± {stats['std']:.4f} (n={stats['n']})"
            if stats["mean"] is not None
            else "—"
        )
        lines.append(f"| {key} | {value} |")
    lines += [
        "",
        "Fidelity expectation: deletion_auc_gradient < deletion_auc_random "
        "and insertion_auc_gradient > insertion_auc_random.",
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    print("evidence:", out_dir, flush=True)


if __name__ == "__main__":
    main()
