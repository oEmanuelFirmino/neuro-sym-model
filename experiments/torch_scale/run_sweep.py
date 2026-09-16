"""Sweep rápido (CPU, escala pequena, minutos) pra achar uma combinação de
`train_frac`/`weight_decay`/`gamma_l1` que mostre a transição
memorização->generalização (grokking) no grafo de conhecimento sintético --
antes de comprometer outra rodada cara de GPU chutando hiperparâmetros.

Mesmo raciocínio de `experiments/modular_addition/run_sweep.py` (que já existe
pra exatamente isso, no domínio de Adição Modular), adaptado pro motor batched
e pro domínio `torch_scale`. Roda os configs em paralelo via joblib -- mesmo
padrão de `experiments/run_multiseed.py`.

Uso: `uv run python experiments/torch_scale/run_sweep.py`
"""

import json
import sys
import time
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from joblib import Parallel, delayed

from experiments.torch_scale.run import build_torch_experiment
from src.neurosym.training.callbacks import Callback

# Escala pequena de propósito -- o objetivo é iterar rápido em hiperparâmetros,
# não treinar o modelo final. A combinação vencedora é escalada depois pro
# treino grande em GPU (experiments/torch_scale/run_pilot.py).
ENTITIES = 30
RELATIONS = 1
ARITY = 3
FACTS_PER_RELATION = 300
EMBEDDING_DIM = 16
HIDDEN = 32
EPOCHS = 20_000
VAL_EVAL_EVERY = 50
SEED = 0

GEN_THRESHOLD = 0.9  # acurácia de validação considerada "generalizou"
GEN_PATIENCE = 3  # avaliações consecutivas acima do limiar

# 1ª rodada (5000 épocas, gamma_l1 em {1e-4, 1e-2}) e 2ª rodada (20000 épocas,
# gamma_l1 mais fino) mostraram o mesmo padrão nas duas: um degrau abrupto
# (gamma_l1 <=3e-4 memoriza sem generalizar; >=1e-3 colapsa pro previsor
# trivial), sem meio-termo, e mais tempo não mudou isso -- 0/38 configs
# generalizaram. Faltava testar a outra metade do método DLG: L1 sozinho não é
# a receita completa, o projeto sempre combina com perda semântica sobre
# axiomas lógicos (ver experiments/modular_addition/axioms.py). 3ª rodada:
# adiciona use_axioms como eixo do sweep pra isolar esse efeito, com gamma_l1
# restrito à região de fronteira (1e-4 a 1e-3, onde os dois regimes se tocam).
TRAIN_FRACS = [0.2, 0.4]
WEIGHT_DECAYS = [0.1, 0.5]
GAMMA_L1S = [1e-4, 3e-4, 1e-3]
USE_AXIOMS = [True, False]

OUTPUT_DIR = Path(__file__).parent / "sweep_results"


class _EpochLogCapture(Callback):
    def __init__(self):
        super().__init__()
        self.epoch_logs: List[Dict] = []

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        self.epoch_logs.append(dict(logs or {}))


def _compute_t_g(epoch_logs: List[Dict], threshold: float, patience: int) -> Optional[int]:
    """Primeira época (1-indexada) a partir da qual val_accuracy fica >=
    threshold por `patience` avaliações consecutivas -- mesma lógica de
    `experiments/run_multiseed.py`."""
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


def _run_config(train_frac: float, weight_decay: float, gamma_l1: float, use_axioms: bool) -> Dict:
    val_frac = min(0.3, (1.0 - train_frac) / 2)

    spec = build_torch_experiment(
        seed=SEED, num_entities=ENTITIES, embedding_dim=EMBEDDING_DIM, hidden=HIDDEN,
        epochs=EPOCHS, num_relations=RELATIONS, arity=ARITY,
        num_positive_facts=FACTS_PER_RELATION, train_frac=train_frac, val_frac=val_frac,
        weight_decay=weight_decay, gamma_l1=gamma_l1, use_axioms=use_axioms, device="cpu",
        val_eval_every=VAL_EVAL_EVERY,
    )
    capture = _EpochLogCapture()
    spec.trainer.callbacks.append(capture)
    capture.set_trainer(spec.trainer)

    t0 = time.time()
    spec.trainer.fit(rules=spec.rules, facts=spec.facts, val_facts=spec.val_facts)
    elapsed = time.time() - t0

    train_acc = spec.trainer.evaluate_accuracy(spec.facts)
    test_acc = spec.trainer.evaluate_accuracy(spec.test_facts)
    t_g = _compute_t_g(capture.epoch_logs, GEN_THRESHOLD, GEN_PATIENCE)
    val_accs = [l["val_accuracy"] for l in capture.epoch_logs if l.get("val_accuracy") is not None]
    peak_val = max(val_accs) if val_accs else None

    return {
        "train_frac": train_frac, "weight_decay": weight_decay, "gamma_l1": gamma_l1,
        "use_axioms": use_axioms, "n_rules": len(spec.rules),
        "train_accuracy": train_acc, "test_accuracy": test_acc, "peak_val_accuracy": peak_val,
        "t_g": t_g, "elapsed_seconds": elapsed,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configs = list(product(TRAIN_FRACS, WEIGHT_DECAYS, GAMMA_L1S, USE_AXIOMS))
    print(f"=== rodando {len(configs)} configs em paralelo (CPU, escala pequena) ===", flush=True)

    t_start = time.time()
    results = Parallel(n_jobs=-1, backend="loky")(
        delayed(_run_config)(*cfg) for cfg in configs
    )
    total_elapsed = time.time() - t_start

    results.sort(key=lambda r: (r["t_g"] is None, r["t_g"] if r["t_g"] is not None else 0))
    for r in results:
        tg_str = f"T_g={r['t_g']}" if r["t_g"] is not None else "não generalizou"
        print(
            f"  train_frac={r['train_frac']} wd={r['weight_decay']} gamma_l1={r['gamma_l1']} "
            f"axiomas={r['use_axioms']}({r['n_rules']}) | "
            f"treino={r['train_accuracy']:.3f} teste={r['test_accuracy']:.3f} "
            f"pico_val={r['peak_val_accuracy']:.3f} | {tg_str} | {r['elapsed_seconds']:.1f}s",
            flush=True,
        )

    with open(OUTPUT_DIR / "grokking_sweep.json", "w", encoding="utf-8") as f:
        json.dump({"results": results, "total_elapsed_seconds": total_elapsed}, f, indent=2)

    winners = [r for r in results if r["t_g"] is not None]
    print(
        f"\n=== concluído em {total_elapsed:.1f}s | {len(winners)}/{len(results)} configs generalizaram ===",
        flush=True,
    )
    print("Resumo salvo em", OUTPUT_DIR / "grokking_sweep.json", flush=True)


if __name__ == "__main__":
    main()
