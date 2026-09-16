"""Benchmark de throughput do motor batched (`torch_engine`), em CPU ou CUDA,
numa escala configurável.

Passo B do plano (obrigatório antes de considerar AWS/GPU): rodar em CPU do piso
atual até a escala-alvo e confirmar que o throughput escala com os FLOPs do
matmul em lote, não mais com a contagem de fatos x overhead Python -- isso prova
que o motor batched já resolve o gargalo principal, independente de GPU.

Passo C (só depois, com aprovação explícita para provisionar AWS): rodar
`--device cuda` na mesma escala numa instância GPU e comparar com o número de
CPU.

Uso:
    uv run python experiments/torch_scale/benchmark.py --device cpu --entities 997 \\
        --embedding-dim 512 --hidden 1024 --num-facts 20000 --epochs 20
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from experiments.torch_scale.run import build_torch_experiment


def main():
    parser = argparse.ArgumentParser(description="Benchmark do motor torch_engine batched")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--entities", type=int, default=200)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--num-facts", type=int, dest="num_positive_facts", default=2000)
    parser.add_argument("--relations", type=int, dest="num_relations", default=1)
    parser.add_argument("--arity", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "--device cuda pedido, mas torch.cuda.is_available() é False nesta máquina."
        )

    spec = build_torch_experiment(
        seed=args.seed,
        num_entities=args.entities,
        embedding_dim=args.embedding_dim,
        hidden=args.hidden,
        epochs=args.epochs,
        num_relations=args.num_relations,
        arity=args.arity,
        num_positive_facts=args.num_positive_facts,
        device=args.device,
    )

    num_train_facts = len(spec.facts)
    print(
        f"device={args.device} entities={args.entities} embedding_dim={args.embedding_dim} "
        f"hidden={args.hidden} train_facts={num_train_facts} epochs={args.epochs}",
        flush=True,
    )

    t0 = time.time()
    spec.trainer.fit(rules=[], facts=spec.facts, val_facts=spec.val_facts)
    if args.device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    epochs_per_sec = args.epochs / elapsed
    facts_per_sec = (num_train_facts * args.epochs) / elapsed
    print(
        f"elapsed={elapsed:.2f}s epochs/s={epochs_per_sec:.2f} facts/s={facts_per_sec:.0f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
