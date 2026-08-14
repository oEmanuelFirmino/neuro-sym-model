"""Busca de hiperparâmetros para destravar a transição de grokking (Fase 3).

A rodada longa (`run_pilot_long.py`, 3000 épocas, wd=1e-2) mostrou um platô de
memorização estável que nunca se rompeu: `l_data`→~0 na época ~300 e mais 2700
épocas sem qualquer subida em `val_accuracy`. Antes de comprometer horas em
p=97, este sweep testa em p=13 as hipóteses documentadas no plano
(docs/plano-correcoes-artigo.md, Fase 3, item 6):

  - weight decay mais forte (0.05 / 0.1 / 0.3) -- na literatura de grokking
    (Power et al. 2022; Nanda et al. 2023) é o botão mais sensível para
    induzir a transição; o default 1e-2 pode ser fraco demais.
  - mais capacidade (embeddings 16, hidden 64) sob wd forte.
  - mais sinal de treino (3 negativos por positivo em vez de 1).

Só o DLG é varrido: a pergunta aqui é "a transição é alcançável neste
framework?", não "quem ganha?". Os baselines voltam quando houver uma
configuração que comprovadamente groka.

Uso: `uv run python experiments/modular_addition/run_sweep.py`
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
from experiments.run_multiseed import run_single_seed

P = 13
EPOCHS = 1500
VAL_EVAL_EVERY = 20
SEED = 0
OUTPUT_DIR = Path(__file__).parent / "sweep_results"

BASE = dict(p=P, epochs=EPOCHS, val_eval_every=VAL_EVAL_EVERY, gamma_l1=1e-4)

CONFIGS = {
    "wd0.05": dict(**BASE, embedding_dim=8, hidden=24, weight_decay=0.05),
    "wd0.1": dict(**BASE, embedding_dim=8, hidden=24, weight_decay=0.1),
    "wd0.3": dict(**BASE, embedding_dim=8, hidden=24, weight_decay=0.3),
    "wd0.1_cap16x64": dict(**BASE, embedding_dim=16, hidden=64, weight_decay=0.1),
    "wd0.1_neg3": dict(
        **BASE, embedding_dim=8, hidden=24, weight_decay=0.1, negatives_per_positive=3
    ),
}


def _peak_val_accuracy(epoch_logs):
    vals = [l["val_accuracy"] for l in epoch_logs if l.get("val_accuracy") is not None]
    return max(vals) if vals else None


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}

    for name, kwargs in CONFIGS.items():
        print(f"=== {name}: {kwargs} ===", flush=True)
        build_fn = make_dlg_build_fn(**kwargs)
        t0 = time.time()
        result = run_single_seed(
            build_fn, seed=SEED, t_g_threshold=0.95, t_g_patience=2
        )
        elapsed = time.time() - t0

        with open(OUTPUT_DIR / f"{name}.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        peak = _peak_val_accuracy(result["epoch_logs"])
        summary[name] = {
            "config": kwargs,
            "t_g": result["t_g"],
            "final_val_accuracy": result["final_val_accuracy"],
            "peak_val_accuracy": peak,
            "test_accuracy": result["test_accuracy"],
            "final_l1_penalty": result["final_l1_penalty"],
            "elapsed_seconds": elapsed,
        }
        print(
            f"  T_g: {result['t_g']} | val final: {result['final_val_accuracy']} | "
            f"val pico: {peak} | teste: {result['test_accuracy']} | {elapsed:.1f}s",
            flush=True,
        )

    with open(OUTPUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nResumo salvo em", OUTPUT_DIR / "summary.json")


if __name__ == "__main__":
    main()
