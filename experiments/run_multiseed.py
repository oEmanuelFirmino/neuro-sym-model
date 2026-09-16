"""Runner de experimento multi-semente, paralelizado por seed.

Cada `build_fn(seed) -> ExperimentSpec` descreve um experimento independente
(embeddings, predicados e split próprios). Rodar N sementes é um problema
"embaraçosamente paralelo" -- não há estado compartilhado entre elas -- então
`run_multiseed` distribui as sementes pelos núcleos de CPU disponíveis via
joblib (backend "loky", que usa cloudpickle e por isso consegue serializar
`build_fn`, que é uma closure retornada por `make_*_build_fn`; o
`multiprocessing`/`pickle` padrão do Python não serializa closures).

Cada processo worker é um interpretador Python novo (spawn), então o backend
de tensor global (`src.neurosym.tensor.backend.current_backend`) NÃO herda o
valor setado no processo principal -- por isso `run_single_seed` seta
`set_backend("numpy")` explicitamente no início, dentro do próprio worker.
"""

from __future__ import annotations

import json
import logging
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from joblib import Parallel, delayed

from src.neurosym.interpreter import Interpreter
from src.neurosym.logic import Formula
from src.neurosym.training.callbacks import Callback
from src.neurosym.training.trainer import Trainer

Fact = Any  # Tuple[Formula, float], mantido solto para não acoplar a um domínio


@dataclass
class ExperimentSpec:
    interpreter: Interpreter
    trainer: Trainer
    rules: List[Formula]
    facts: List[Fact]
    val_facts: Optional[List[Fact]] = None
    test_facts: Optional[List[Fact]] = None


class _EpochLogCapture(Callback):
    def __init__(self):
        super().__init__()
        self.epoch_logs: List[Dict] = []

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        self.epoch_logs.append(dict(logs or {}))


def _compute_t_g(
    epoch_logs: List[Dict], threshold: float, patience: int
) -> Optional[int]:
    """Primeira época (1-indexada) a partir da qual val_accuracy fica >=
    threshold por `patience` avaliações consecutivas (transição de grokking).
    Retorna None se a transição nunca ocorre no treino."""
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


def run_single_seed(
    build_fn: Callable[[int], ExperimentSpec],
    seed: int,
    t_g_threshold: float = 0.95,
    t_g_patience: int = 2,
) -> Dict[str, Any]:
    from src.neurosym.tensor.backend import set_backend

    # Cada seed roda num processo worker novo (spawn): nem o backend global
    # nem `logging.disable` do processo principal são herdados, então ambos
    # precisam ser refeitos aqui dentro -- caso contrário o worker volta ao
    # PythonBackend (mais lento) e o Trainer volta a logar por época,
    # interlaçando saída de vários processos e (no console cp1252 do
    # Windows) derrubando o handler ao tentar imprimir os emojis do logger.
    set_backend("numpy")
    logging.disable(logging.CRITICAL)

    spec = build_fn(seed)

    capture = _EpochLogCapture()
    spec.trainer.callbacks.append(capture)
    capture.set_trainer(spec.trainer)

    t0 = time.time()
    spec.trainer.fit(rules=spec.rules, facts=spec.facts, val_facts=spec.val_facts)
    elapsed = time.time() - t0

    final_val_accuracy = (
        capture.epoch_logs[-1].get("val_accuracy") if capture.epoch_logs else None
    )
    final_l1_penalty = (
        capture.epoch_logs[-1].get("l1_penalty") if capture.epoch_logs else None
    )
    test_accuracy = (
        spec.trainer.evaluate_accuracy(spec.test_facts) if spec.test_facts else None
    )

    return {
        "seed": seed,
        "t_g": _compute_t_g(capture.epoch_logs, t_g_threshold, t_g_patience),
        "final_val_accuracy": final_val_accuracy,
        "test_accuracy": test_accuracy,
        "final_l1_penalty": final_l1_penalty,
        "elapsed_seconds": elapsed,
        "epoch_logs": capture.epoch_logs,
    }


def _agg_mean(results: List[Dict[str, Any]], key: str) -> Optional[float]:
    values = [r[key] for r in results if r.get(key) is not None]
    return statistics.fmean(values) if values else None


def run_multiseed(
    build_fn: Callable[[int], ExperimentSpec],
    seeds: List[int],
    t_g_threshold: float = 0.95,
    t_g_patience: int = 2,
    output_path: Optional[str] = None,
    n_jobs: int = -1,
) -> Dict[str, Any]:
    """Roda `build_fn` para cada seed em paralelo (um processo por seed, até
    `n_jobs` de concorrência; -1 usa todos os núcleos disponíveis) e agrega os
    resultados por média simples entre sementes."""
    t_start = time.time()
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(run_single_seed)(build_fn, seed, t_g_threshold, t_g_patience)
        for seed in seeds
    )
    elapsed = time.time() - t_start

    aggregate = {
        "t_g": _agg_mean(results, "t_g"),
        "final_val_accuracy": _agg_mean(results, "final_val_accuracy"),
        "test_accuracy": _agg_mean(results, "test_accuracy"),
        "final_l1_penalty": _agg_mean(results, "final_l1_penalty"),
    }

    payload = {
        "seeds": seeds,
        "aggregate": aggregate,
        "runs": [{k: v for k, v in r.items() if k != "epoch_logs"} for r in results],
        "elapsed_seconds": elapsed,
    }

    if output_path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    return payload
