"""Experimento de grokking na formulação por classificação softmax (Fase 3).

Compara duas variantes em p=13, 3000 épocas:
  - `softmax_dlg`: com axiomas distribucionais (comutatividade + identidade),
    o análogo da proposta DLG nesta formulação;
  - `softmax_mlp`: classificador puro, sem axiomas — o baseline.

weight_decay=1.0 (canônico da literatura de grokking para classificação).
Evidência (tabelas + curvas + comparação) gerada automaticamente via
`experiments/reporting.py`.

Uso: `uv run python experiments/modular_addition/run_softmax.py`
"""

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.neurosym.tensor.backend import set_backend

set_backend("numpy")
logging.disable(logging.CRITICAL)

from experiments.modular_addition.softmax_formulation import make_softmax_build_fn
from experiments.reporting import save_comparison_report, save_run_report
from experiments.run_multiseed import run_multiseed

P = 13
EPOCHS = 3000
VAL_EVAL_EVERY = 10
SEEDS = [0, 1]

COMMON = dict(
    p=P,
    embedding_dim=16,
    hidden=48,
    epochs=EPOCHS,
    weight_decay=1.0,
    val_eval_every=VAL_EVAL_EVERY,
)

CONFIGS = {
    "softmax_dlg": dict(**COMMON, use_axioms=True),
    "softmax_mlp": dict(**COMMON, use_axioms=False),
}


def main():
    results = {}
    for name, kwargs in CONFIGS.items():
        print(f"=== {name}: {kwargs} ===", flush=True)
        build_fn = make_softmax_build_fn(**kwargs)
        t0 = time.time()
        result = run_multiseed(
            build_fn, seeds=SEEDS, t_g_threshold=0.95, t_g_patience=2
        )
        elapsed = time.time() - t0
        results[name] = result
        agg = result["aggregate"]
        print(
            f"  T_g: {agg['t_g']} | val: {agg['final_val_accuracy']} | "
            f"teste: {agg['test_accuracy']} | {elapsed:.1f}s",
            flush=True,
        )
        evidence = save_run_report(name, result, config=kwargs, timestamp=False)
        print("  evidence:", evidence, flush=True)

    comparison = save_comparison_report(
        "softmax_comparison", results, configs=CONFIGS, timestamp=False
    )
    print("evidence (comparison):", comparison, flush=True)


if __name__ == "__main__":
    main()
