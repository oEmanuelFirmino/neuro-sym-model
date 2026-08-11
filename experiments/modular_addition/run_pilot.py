"""Piloto em pequena escala do domínio de Adição Modular -- Fase 3 do plano
de correções do artigo.

NÃO é a escala do artigo (p=97, 10 seeds, milhares de épocas). É uma
validação honesta do pipeline (dataset -> split -> treino -> T_g -> baselines)
em escala reduzida, para confirmar que a infraestrutura funciona de ponta a
ponta e dar um primeiro sinal (não uma conclusão) sobre se o DLG generaliza
mais rápido que os baselines. A escala completa exige uma execução muito mais
longa (ver docs/plano-correcoes-artigo.md, Fase 3).

Uso: `uv run python experiments/modular_addition/run_pilot.py`
"""

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.neurosym.tensor.backend import set_backend

set_backend("numpy")
logging.disable(logging.CRITICAL)  # evita o UnicodeEncodeError de emojis no console cp1252 do Windows

from experiments.modular_addition.run import (
    make_dlg_build_fn,
    make_ltn_baseline_build_fn,
    make_mlp_baseline_build_fn,
)
from experiments.run_multiseed import run_multiseed

P = 13
EMBEDDING_DIM = 8
HIDDEN = 24
EPOCHS = 200
VAL_EVAL_EVERY = 5
SEEDS = [0, 1]
OUTPUT_DIR = Path(__file__).parent / "pilot_results"

ARCHITECTURES = {
    "dlg": make_dlg_build_fn(
        p=P,
        embedding_dim=EMBEDDING_DIM,
        hidden=HIDDEN,
        epochs=EPOCHS,
        val_eval_every=VAL_EVAL_EVERY,
        gamma_l1=1e-4,
    ),
    "mlp_baseline": make_mlp_baseline_build_fn(
        p=P,
        embedding_dim=EMBEDDING_DIM,
        hidden=HIDDEN,
        epochs=EPOCHS,
        val_eval_every=VAL_EVAL_EVERY,
    ),
    "ltn_baseline": make_ltn_baseline_build_fn(
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

    with open(OUTPUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nResumo salvo em", OUTPUT_DIR / "summary.json")


if __name__ == "__main__":
    main()
