"""Fidelidade axiomática (Tabela 4 real) + recalibração de gamma_l1 (esparsidade).

Parte A — fidelidade: treina DLG / MLP puro / LTN em p=13 (formulação
relacional, onde o mecanismo de inferência existe) e mede a satisfação dos
axiomas em instanciações held-out (pares de validação) — a coluna
"Fidelidade (%)" da Tabela 4 do artigo, medida de verdade.

Parte B — esparsidade: varre gamma_l1 ∈ {1e-4, 1e-3, 1e-2, 1e-1} no DLG e mede
a fração de pesos |w| < 1e-3 (métrica comparável entre arquiteturas, item m3)
junto com a fidelidade — para achar o gamma que produz esparsidade genuína sem
destruir a consistência lógica.

Uso: `uv run python experiments/modular_addition/run_fidelity_sparsity.py`
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

import random

import numpy as np

from experiments.modular_addition.dataset import generate_split
from experiments.modular_addition.fidelity import axiom_fidelity
from experiments.modular_addition.run import (
    make_dlg_build_fn,
    make_ltn_baseline_build_fn,
    make_mlp_baseline_build_fn,
)
from experiments.reporting import EVIDENCE_ROOT
from src.neurosym.training.metrics import weight_sparsity

P = 13
EPOCHS = 400
SEED = 0
COMMON = dict(p=P, embedding_dim=8, hidden=24, epochs=EPOCHS, val_eval_every=50)


def _train(build_fn):
    random.seed(SEED)
    np.random.seed(SEED)
    spec = build_fn(SEED)
    t0 = time.time()
    spec.trainer.fit(rules=spec.rules, facts=spec.facts)
    return spec, time.time() - t0


def _evaluate(spec):
    data = generate_split(P, seed=SEED)
    fidelity = axiom_fidelity(spec.interpreter, data.val_pairs, P)
    weights = []
    for model in spec.interpreter.predicate_map.values():
        weights.extend(model.l1_weight_parameters())
    sparsity = weight_sparsity(weights, eps=1e-3)
    return fidelity, sparsity


def main():
    out_dir = EVIDENCE_ROOT / "fidelity_sparsity_p13"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"config": COMMON, "part_a_fidelity": {}, "part_b_gamma_sweep": {}}

    print("=== Parte A: fidelidade axiomática (DLG / MLP / LTN) ===", flush=True)
    architectures = {
        "dlg": make_dlg_build_fn(**COMMON, gamma_l1=1e-4),
        "mlp_baseline": make_mlp_baseline_build_fn(**COMMON),
        "ltn_baseline": make_ltn_baseline_build_fn(**COMMON),
    }
    for name, build_fn in architectures.items():
        spec, elapsed = _train(build_fn)
        fidelity, sparsity = _evaluate(spec)
        report["part_a_fidelity"][name] = {
            "fidelity": fidelity,
            "sparsity": sparsity,
            "train_seconds": elapsed,
        }
        print(
            f"  {name}: fidelidade geral {fidelity['overall']['mean']:.4f} "
            f"(comut {fidelity['commutativity']['mean']:.4f}, "
            f"ident {fidelity['identity']['mean']:.4f}) | "
            f"esparsidade {sparsity:.4f} | {elapsed:.0f}s",
            flush=True,
        )

    print("=== Parte B: varredura de gamma_l1 (DLG) ===", flush=True)
    for gamma in (1e-4, 1e-3, 1e-2, 1e-1):
        spec, elapsed = _train(make_dlg_build_fn(**COMMON, gamma_l1=gamma))
        fidelity, sparsity = _evaluate(spec)
        report["part_b_gamma_sweep"][f"gamma_{gamma:g}"] = {
            "gamma_l1": gamma,
            "fidelity_overall": fidelity["overall"],
            "sparsity": sparsity,
            "train_seconds": elapsed,
        }
        print(
            f"  gamma={gamma:g}: esparsidade {sparsity:.4f} | "
            f"fidelidade {fidelity['overall']['mean']:.4f} | {elapsed:.0f}s",
            flush=True,
        )

    with open(out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # resumo markdown
    lines = [
        "# Axiom fidelity (held-out) + gamma_l1 sweep — p=13",
        "",
        "## Part A — fidelity by architecture",
        "",
        "| Architecture | Overall fidelity | Commutativity | Identity | Sparsity |",
        "|---|---|---|---|---|",
    ]
    for name, entry in report["part_a_fidelity"].items():
        fid = entry["fidelity"]
        lines.append(
            f"| {name} | {fid['overall']['mean']:.4f} | "
            f"{fid['commutativity']['mean']:.4f} | {fid['identity']['mean']:.4f} | "
            f"{entry['sparsity']:.4f} |"
        )
    lines += [
        "",
        "## Part B — gamma_l1 sweep (DLG)",
        "",
        "| gamma_l1 | Sparsity (|w| < 1e-3) | Overall fidelity |",
        "|---|---|---|",
    ]
    for key, entry in report["part_b_gamma_sweep"].items():
        lines.append(
            f"| {entry['gamma_l1']:g} | {entry['sparsity']:.4f} | "
            f"{entry['fidelity_overall']['mean']:.4f} |"
        )
    lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("evidence:", out_dir, flush=True)


if __name__ == "__main__":
    main()
