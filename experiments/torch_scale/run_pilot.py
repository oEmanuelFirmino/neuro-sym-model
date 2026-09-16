"""Treino real (não benchmark de throughput) sobre o grafo de conhecimento
sintético (`experiments/torch_scale/dataset.py`): várias relações, muitas
entidades, dezenas/centenas de milhares de fatos, `embedding_dim`/`hidden`
grandes -- a escala que só faz sentido em GPU (validado nesta sessão: ~33x mais
rápido que CPU na mesma máquina, na configuração embedding_dim=1024/hidden=2048).

Uso (dentro do notebook do Colab, GPU ativa):
    !python experiments/torch_scale/run_pilot.py --device cuda
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from experiments.torch_scale.run import build_torch_experiment
from src.neurosym.torch_engine import save_model
from src.neurosym.training.callbacks import Callback

OUTPUT_DIR = Path(__file__).parent / "pilot_results"

GEN_THRESHOLD = 0.9
GEN_PATIENCE = 3


class _EpochLogCapture(Callback):
    def __init__(self):
        super().__init__()
        self.epoch_logs = []

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        self.epoch_logs.append(dict(logs or {}))


def _compute_t_g(epoch_logs, threshold, patience):
    """Primeira época (1-indexada) a partir da qual val_accuracy fica >=
    threshold por `patience` avaliações consecutivas -- mesma lógica de
    `experiments/torch_scale/run_sweep.py`/`experiments/run_multiseed.py`."""
    streak = 0
    for i, log in enumerate(epoch_logs):
        acc = log.get("val_accuracy")
        if acc is not None and acc >= threshold:
            streak += 1
            if streak >= patience:
                return i - patience + 2
        else:
            streak = 0
    return None


def main():
    parser = argparse.ArgumentParser(description="Treino real do grafo de conhecimento sintético")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--entities", type=int, default=2000)
    parser.add_argument("--relations", type=int, dest="num_relations", default=5)
    parser.add_argument("--arity", type=int, default=3)
    parser.add_argument("--facts-per-relation", type=int, dest="num_positive_facts", default=20_000)
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--val-eval-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--weight-decay", type=float, default=0.1,
        help="AdamW weight decay -- o botão mais sensível pra induzir a transição "
        "memorização->generalização (grokking); 1e-2 (padrão do AdamW) não foi "
        "suficiente na primeira rodada (treino=1.0, teste=0.50).",
    )
    parser.add_argument(
        "--gamma-l1", type=float, default=1e-4,
        help="Peso da penalidade L1 estrutural do DLG (||W||_1). Estava em 0.0 na "
        "primeira rodada -- o mecanismo central da abordagem nem foi exercido.",
    )
    parser.add_argument("--train-frac", type=float, default=0.4)
    parser.add_argument("--val-frac", type=float, default=0.3)
    parser.add_argument(
        "--no-axioms", dest="use_axioms", action="store_false",
        help="Desliga os axiomas de comutatividade (perda semântica). Ligado por "
        "padrão -- o sweep de hiperparâmetros (CPU, escala pequena) mostrou que "
        "L1 sozinho, sem axiomas, nunca generalizou em 62 configs testadas.",
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda pedido, mas torch.cuda.is_available() é False nesta máquina.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
        val_eval_every=args.val_eval_every,
        lr=args.lr,
        weight_decay=args.weight_decay,
        gamma_l1=args.gamma_l1,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        use_axioms=args.use_axioms,
    )
    capture = _EpochLogCapture()
    spec.trainer.callbacks.append(capture)
    capture.set_trainer(spec.trainer)

    print(
        f"device={args.device} entities={args.entities} relations={args.num_relations} "
        f"arity={args.arity} embedding_dim={args.embedding_dim} hidden={args.hidden} "
        f"train_facts={len(spec.facts)} val_facts={len(spec.val_facts)} "
        f"test_facts={len(spec.test_facts)} rules={len(spec.rules)} epochs={args.epochs}",
        flush=True,
    )

    t0 = time.time()
    spec.trainer.fit(rules=spec.rules, facts=spec.facts, val_facts=spec.val_facts)
    if args.device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    train_acc = spec.trainer.evaluate_accuracy(spec.facts)
    test_acc = spec.trainer.evaluate_accuracy(spec.test_facts)
    gap = train_acc - test_acc
    t_g = _compute_t_g(capture.epoch_logs, GEN_THRESHOLD, GEN_PATIENCE)
    val_accs = [l["val_accuracy"] for l in capture.epoch_logs if l.get("val_accuracy") is not None]
    peak_val = max(val_accs) if val_accs else None
    print(
        f"\nTreino concluído em {elapsed:.1f}s. "
        f"Acurácia -- treino: {train_acc:.4f} | teste: {test_acc:.4f} | gap: {gap:.4f} | "
        f"pico_val: {peak_val:.4f} | T_g: {t_g}",
        flush=True,
    )
    if gap > 0.3:
        print(
            "AVISO: gap grande entre treino e teste -- ainda memorizando sem "
            "generalizar. Pode precisar de mais épocas e/ou weight_decay maior.",
            flush=True,
        )

    model_path = OUTPUT_DIR / f"torch_scale_kg_{args.device}.pt"
    save_model(str(model_path), spec.model)
    print(f"Modelo salvo em {model_path}", flush=True)

    summary = {
        "config": vars(args),
        "train_facts": len(spec.facts),
        "val_facts": len(spec.val_facts),
        "test_facts": len(spec.test_facts),
        "elapsed_seconds": elapsed,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "train_test_gap": gap,
        "peak_val_accuracy": peak_val,
        "t_g": t_g,
        "val_accuracy_curve": val_accs,
        "model_path": str(model_path),
    }
    summary_path = OUTPUT_DIR / f"torch_scale_kg_{args.device}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Resumo salvo em {summary_path}", flush=True)


if __name__ == "__main__":
    main()
