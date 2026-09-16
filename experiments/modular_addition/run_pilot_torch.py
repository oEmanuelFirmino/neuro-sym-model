"""Treino real (não só benchmark de throughput) de Adição Modular no motor
batched, numa escala bem mais próxima de um cenário real que o piloto original
(p=13, hidden=16-64): p=197, embedding_dim=64, hidden=128 -- ~38.8 mil fatos de
treino. Split generoso (train_frac=0.5) porque o objetivo aqui é um modelo
utilizável pra inferência depois, não estudar a transição de grokking (que pede
justamente escassez de dados de treino).

Uso: `uv run python experiments/modular_addition/run_pilot_torch.py`
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.modular_addition.run_torch import (
    build_torch_modular_addition_experiment,
    predict_add,
)
from src.neurosym.torch_engine import save_model

P = 197
EMBEDDING_DIM = 64
HIDDEN = 128
EPOCHS = 1000
VAL_EVAL_EVERY = 25
SEED = 0
OUTPUT_DIR = Path(__file__).parent / "pilot_results"
MODEL_PATH = OUTPUT_DIR / "modular_addition_torch_p197.pt"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    spec = build_torch_modular_addition_experiment(
        seed=SEED, p=P, embedding_dim=EMBEDDING_DIM, hidden=HIDDEN, epochs=EPOCHS,
        train_frac=0.5, val_frac=0.25, val_eval_every=VAL_EVAL_EVERY,
    )
    print(
        f"p={P} embedding_dim={EMBEDDING_DIM} hidden={HIDDEN} epochs={EPOCHS} "
        f"train_facts={len(spec.facts)} val_facts={len(spec.val_facts)} "
        f"test_facts={len(spec.test_facts)} rules={len(spec.rules)}",
        flush=True,
    )

    t0 = time.time()
    spec.trainer.fit(rules=spec.rules, facts=spec.facts, val_facts=spec.val_facts)
    elapsed = time.time() - t0

    test_acc = spec.trainer.evaluate_accuracy(spec.test_facts)
    print(f"\nTreino concluído em {elapsed:.1f}s. Acurácia (argmax) no teste: {test_acc:.4f}", flush=True)

    save_model(str(MODEL_PATH), spec.model)
    print(f"Modelo salvo em {MODEL_PATH}", flush=True)

    examples = [(3, 5), (10, 10), (100, 150), (196, 1), (50, 50)]
    print("\nExemplos de inferência (Add(a,b) -> c previsto vs. correto):", flush=True)
    predictions = []
    for a, b in examples:
        pred = predict_add(spec.model, spec.trainer.interpreter, a, b, P)
        correct = (a + b) % P
        status = "OK" if pred == correct else "ERRO"
        print(f"  Add({a},{b}) -> previsto {pred}, correto {correct} [{status}]", flush=True)
        predictions.append({"a": a, "b": b, "predicted": pred, "correct": correct, "status": status})

    summary = {
        "config": {
            "p": P, "embedding_dim": EMBEDDING_DIM, "hidden": HIDDEN, "epochs": EPOCHS,
            "train_frac": 0.5, "val_frac": 0.25, "seed": SEED,
        },
        "train_facts": len(spec.facts), "val_facts": len(spec.val_facts),
        "test_facts": len(spec.test_facts), "elapsed_seconds": elapsed,
        "test_accuracy": test_acc, "example_predictions": predictions,
        "model_path": str(MODEL_PATH),
    }
    with open(OUTPUT_DIR / "modular_addition_torch_p197_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResumo salvo em {OUTPUT_DIR / 'modular_addition_torch_p197_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
