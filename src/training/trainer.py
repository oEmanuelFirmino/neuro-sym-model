import sys
import logging
from typing import List, Tuple
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.tensor import Tensor
from src.logic import Formula
from src.interpreter import Interpreter
from src.training.optimizer import SGD


class Trainer:
    def __init__(self, interpreter: Interpreter, optimizer: SGD, epochs: int):
        self.interpreter = interpreter
        self.optimizer = optimizer
        self.epochs = epochs
        self.logger = self._setup_logger()

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

        for epoch in range(self.epochs):
            self.optimizer.zero_grad()

            rule_truth_values = [
                self.interpreter.eval_formula(formula, {}) for formula in rules
            ]
            fact_losses = [
                (
                    self.interpreter.eval_formula(fact_formula, {}).sum()
                    - Tensor(truth_value)
                )
                ** 2
                for fact_formula, truth_value in facts
            ]

            total_satisfaction = sum(rule_truth_values, Tensor(0.0))
            total_fact_loss = sum(fact_losses, Tensor(0.0))
            loss = (Tensor(len(rules)) - total_satisfaction) + total_fact_loss

            loss.backward()
            self.optimizer.step()

            if (epoch + 1) % 10 == 0 or epoch == self.epochs - 1:
                avg_satisfaction = (
                    (total_satisfaction.data / len(rules)) if rules else 1.0
                )
                self.logger.info(
                    f"Época [{epoch+1}/{self.epochs}], Perda: {loss.data:.4f}, Satisfação das Regras: {avg_satisfaction:.4f}"
                )

        self.logger.info("--- ✅ Treinamento Concluído ---")
