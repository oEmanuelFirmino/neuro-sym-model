import sys
import logging
from typing import List, Tuple, Dict
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

from src.neurosym.tensor import Tensor
from src.neurosym.logic import Formula
from src.neurosym.interpreter import Interpreter
from src.neurosym.training.optimizer import SGD
from src.neurosym.training.callbacks import Callback


class Trainer:
    def __init__(
        self,
        interpreter: Interpreter,
        optimizer: SGD,
        epochs: int,
        callbacks: List[Callback] = None,
    ):
        self.interpreter = interpreter
        self.optimizer = optimizer
        self.epochs = epochs
        self.callbacks = callbacks if callbacks else []
        self.logger = self._setup_logger()

        for cb in self.callbacks:
            cb.set_trainer(self)

    def _setup_logger(self):
        logger = logging.getLogger("NeuroSymbolicTrainer")
        logger.setLevel(logging.INFO)
        if not logger.hasHandlers():
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                "%(asctime)s | %(name)s | %(message)s", datefmt="%H:%M:%S"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def fit(self, rules: List[Formula], facts: List[Tuple[Formula, float]]):
        self.logger.info("--- 🚀 Iniciando Loop de Treinamento 🚀 ---")
        logs: Dict[str, float] = {}

        self.callbacks_handler("on_train_begin", logs)

        for epoch in range(self.epochs):
            self.callbacks_handler("on_epoch_begin", epoch, logs)

            self.optimizer.zero_grad()

            rule_truth_values = [self.interpreter.eval_formula(r, {}) for r in rules]
            fact_losses = [
                (self.interpreter.eval_formula(f, {}).sum() - Tensor(t)) ** 2
                for f, t in facts
            ]

            total_satisfaction = (
                sum(rule_truth_values, Tensor(0.0)) if rules else Tensor(0.0)
            )
            total_fact_loss = sum(fact_losses, Tensor(0.0)) if facts else Tensor(0.0)

            loss = (Tensor(len(rules)) - total_satisfaction) + total_fact_loss

            loss.backward()
            self.optimizer.step()

            avg_satisfaction = (total_satisfaction.data / len(rules)) if rules else 1.0
            logs["loss"] = loss.data
            logs["satisfaction"] = avg_satisfaction

            if (epoch + 1) % 10 == 0 or epoch == self.epochs - 1:
                self.logger.info(
                    f"Época [{epoch+1}/{self.epochs}], Perda: {logs['loss']:.4f}, Satisfação das Regras: {logs['satisfaction']:.4f}"
                )

            self.callbacks_handler("on_epoch_end", epoch, logs)

        self.callbacks_handler("on_train_end", logs)
        self.logger.info("--- ✅ Treinamento Concluído ---")

    def callbacks_handler(self, method_name: str, *args):
        for cb in self.callbacks:
            method = getattr(cb, method_name)
            method(*args)
