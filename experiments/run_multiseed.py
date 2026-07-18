"""Harness de execução multi-seed para experimentos DLG.

Resolve o item Q6 do parecer do orientador (número de execuções divergente
entre texto/Figura 6/Figura 8: 10/5/50) ao centralizar a execução multi-seed
em um único ponto: todo experimento reportado no artigo deve passar por este
runner, com o mesmo N de seeds em todo lugar.

O runner é propositalmente agnóstico de domínio: recebe um `build_fn(seed)`
que constrói e devolve um `ExperimentSpec` (interpreter, trainer, regras,
fatos de treino/validação/teste já ligados a essa seed) e apenas orquestra a
repetição, a agregação (média/desvio padrão) e a serialização em JSON. O
domínio de Adição Modular (Fase 3 do plano de correções) usará este mesmo
runner passando seu próprio `build_fn`.
"""

import json
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from src.neurosym.interpreter import Interpreter
from src.neurosym.logic import Formula
from src.neurosym.training.callbacks import Callback
from src.neurosym.training.metrics import time_to_generalization
from src.neurosym.training.trainer import Trainer


@dataclass
class ExperimentSpec:
    interpreter: Interpreter
    trainer: Trainer
    rules: List[Formula]
    facts: List[Tuple[Formula, float]]
    val_facts: Optional[List[Tuple[Formula, float]]] = None
    test_facts: Optional[List[Tuple[Formula, float]]] = None


BuildFn = Callable[[int], ExperimentSpec]


class _CurveCapture(Callback):
    """Callback interno que só acumula os logs de cada época para pós-processo."""

    def __init__(self):
        super().__init__()
        self.epoch_logs: List[Dict] = []

    def on_epoch_end(self, epoch, logs: Dict = None):
        super().on_epoch_end(epoch, logs)
        self.epoch_logs.append(dict(logs or {}))


def _seed_everything(seed: int):
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


def run_single_seed(
    build_fn: BuildFn,
    seed: int,
    t_g_threshold: float = 0.95,
    t_g_patience: int = 1,
) -> Dict:
    """Roda um experimento completo para uma única seed e devolve suas métricas."""
    _seed_everything(seed)
    spec = build_fn(seed)

    capture = _CurveCapture()
    spec.trainer.callbacks.append(capture)
    capture.set_trainer(spec.trainer)

    spec.trainer.fit(rules=spec.rules, facts=spec.facts, val_facts=spec.val_facts)

    val_curve = [
        epoch["val_accuracy"]
        for epoch in capture.epoch_logs
        if epoch.get("val_accuracy") is not None
    ]

    t_g = (
        time_to_generalization(val_curve, threshold=t_g_threshold, patience=t_g_patience)
        if val_curve
        else None
    )
    test_accuracy = (
        spec.trainer.evaluate_accuracy(spec.test_facts) if spec.test_facts else None
    )
    final_val_accuracy = val_curve[-1] if val_curve else None
    final_l1_penalty = (
        capture.epoch_logs[-1].get("l1_penalty") if capture.epoch_logs else None
    )

    return {
        "seed": seed,
        "t_g": t_g,
        "final_val_accuracy": final_val_accuracy,
        "test_accuracy": test_accuracy,
        "final_l1_penalty": final_l1_penalty,
        "epoch_logs": capture.epoch_logs,
    }


def _mean_std(values: List[Optional[float]]) -> Dict[str, Optional[float]]:
    clean = [v for v in values if v is not None]
    if not clean:
        return {"mean": None, "std": None, "n": 0}
    mean = statistics.fmean(clean)
    std = statistics.pstdev(clean) if len(clean) > 1 else 0.0
    return {"mean": mean, "std": std, "n": len(clean)}


def run_multiseed(
    build_fn: BuildFn,
    seeds: Optional[List[int]] = None,
    n_seeds: int = 10,
    t_g_threshold: float = 0.95,
    t_g_patience: int = 1,
    output_path: Optional[str] = None,
) -> Dict:
    """Roda `build_fn` para cada seed em `seeds` (padrão: 0..n_seeds-1),
    agrega os resultados (média +/- desvio padrão) e, se `output_path` for
    informado, serializa tudo (agregados e curvas por época de cada seed) em
    JSON -- insumo bruto para Tabelas 4/5 e Figuras 6-8 do artigo.
    """
    seeds = seeds if seeds is not None else list(range(n_seeds))

    runs = [
        run_single_seed(
            build_fn, seed, t_g_threshold=t_g_threshold, t_g_patience=t_g_patience
        )
        for seed in seeds
    ]

    aggregate = {
        "n_seeds": len(seeds),
        "seeds": seeds,
        "t_g": _mean_std([r["t_g"] for r in runs]),
        "final_val_accuracy": _mean_std([r["final_val_accuracy"] for r in runs]),
        "test_accuracy": _mean_std([r["test_accuracy"] for r in runs]),
        "final_l1_penalty": _mean_std([r["final_l1_penalty"] for r in runs]),
    }

    result = {"aggregate": aggregate, "runs": runs}

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    return result
