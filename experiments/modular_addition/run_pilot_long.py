"""Rodada mais longa (DLG vs MLP puro) do domínio de Adição Modular.

O piloto curto (`run_pilot.py`, 200 épocas) validou o pipeline mas foi tempo
demais curto para observar a transição de grokking em si -- a curva mostrou
um platô de memorização (loss caindo, val_accuracy estagnada) sem cruzar o
limiar de generalização. Este script roda só as duas arquiteturas mais
informativas (DLG vs MLP puro, sem LTN) por muito mais épocas para dar um
primeiro sinal real sobre a hipótese central do artigo: o DLG generaliza mais
rápido que um baseline puramente estatístico?

Ainda não é a escala do artigo (p=97, 10 seeds) -- ver docs/plano-correcoes-artigo.md.

Uso: `uv run python experiments/modular_addition/run_pilot_long.py`
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

from experiments.modular_addition.run import make_dlg_build_fn, make_mlp_baseline_build_fn
from experiments.run_multiseed import run_multiseed

P = 13
EMBEDDING_DIM = 8
HIDDEN = 24
EPOCHS = 3000
VAL_EVAL_EVERY = 20
SEEDS = [0, 1]
OUTPUT_DIR = Path(__file__).parent / "pilot_results"

ARCHITECTURES = {
    "dlg_long": make_dlg_build_fn(
        p=P,
        embedding_dim=EMBEDDING_DIM,
        hidden=HIDDEN,
        epochs=EPOCHS,
        val_eval_every=VAL_EVAL_EVERY,
        gamma_l1=1e-4,
    ),
    "mlp_baseline_long": make_mlp_baseline_build_fn(
        p=P,
        embedding_dim=EMBEDDING_DIM,
        hidden=HIDDEN,
        epochs=EPOCHS,
        val_eval_every=VAL_EVAL_EVERY,
    ),
}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}

    for name, build_fn in ARCHITECTURES.items():
        print(f"=== {name} (p={P}, epochs={EPOCHS}, seeds={SEEDS}) ===", flush=True)
        t0 = time.time()
        result = run_multiseed(
            build_fn,
            seeds=SEEDS,
            t_g_threshold=0.95,
            t_g_patience=2,
            output_path=str(OUTPUT_DIR / f"{name}.json"),
        )
        elapsed = time.time() - t0
        agg = result["aggregate"]
        summary[name] = {**agg, "elapsed_seconds": elapsed}
        print(
            f"  T_g: {agg['t_g']} | val_acc: {agg['final_val_accuracy']} | "
            f"test_acc: {agg['test_accuracy']} | l1_penalty: {agg['final_l1_penalty']} | "
            f"{elapsed:.1f}s",
            flush=True,
        )

    with open(OUTPUT_DIR / "summary_long.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nResumo salvo em", OUTPUT_DIR / "summary_long.json")


if __name__ == "__main__":
    main()
