import sys
import logging
from typing import List, Tuple, Dict, Optional
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
        lambda_semantic: float = 1.0,
        gamma_l1: float = 0.0,
        val_threshold: float = 0.5,
    ):
        """
        lambda_semantic e gamma_l1 correspondem aos pesos lambda e gamma do
        funcional L_total = L_data + lambda*L_semantic + gamma*||W||_1
        (artigo, Seção 4.4). gamma_l1=0.0 por padrão desativa a regularização
        estrutural, preservando o comportamento anterior por padrão.
        """
        self.interpreter = interpreter
        self.optimizer = optimizer
        self.epochs = epochs
        self.callbacks = callbacks if callbacks else []
        self.lambda_semantic = lambda_semantic
        self.gamma_l1 = gamma_l1
        self.val_threshold = val_threshold
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

    def _l1_penalty(self) -> Tensor:
        weight_tensors = []
        for model in self.interpreter.predicate_map.values():
            weight_tensors.extend(model.l1_weight_parameters())

        if not weight_tensors:
            return Tensor(0.0)

        return sum((w.abs().sum() for w in weight_tensors), Tensor(0.0))

    def evaluate_accuracy(
        self, facts: List[Tuple[Formula, float]], threshold: Optional[float] = None
    ) -> Optional[float]:
        """Fração de fatos cujo grau de verdade previsto cai do mesmo lado do
        limiar que o valor-alvo (acurácia binária), sem calcular gradientes.
        Usada para acompanhar a curva de validação por época (necessária para
        T_g e para as Figuras de grokking) e para avaliação final em holdout.
        """
        if not facts:
            return None

        threshold = self.val_threshold if threshold is None else threshold
        correct = 0
        for formula, target in facts:
            predicted = self.interpreter.eval_formula(formula, {})
            predicted_value = predicted._flatten(predicted.data)[0]
            if (predicted_value >= threshold) == (target >= threshold):
                correct += 1
        return correct / len(facts)

    def fit(
        self,
        rules: List[Formula],
        facts: List[Tuple[Formula, float]],
        val_facts: Optional[List[Tuple[Formula, float]]] = None,
    ):
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

            l_data = (
                (sum(fact_losses, Tensor(0.0)) / len(facts))
                if facts
                else Tensor(0.0)
            )

            if rules:
                avg_satisfaction = sum(rule_truth_values, Tensor(0.0)) / len(rules)
                l_semantic = Tensor(1.0) - avg_satisfaction
            else:
                avg_satisfaction = Tensor(1.0)
                l_semantic = Tensor(0.0)

            # Sempre computado (mesmo com gamma_l1=0) para permitir monitorar a
            # evolução da esparsidade nos logs independentemente do peso aplicado.
            l1_penalty = self._l1_penalty()

            loss = (
                l_data
                + Tensor(self.lambda_semantic) * l_semantic
                + Tensor(self.gamma_l1) * l1_penalty
            )

            loss.backward()
            self.optimizer.step()

            logs["loss"] = loss.data
            logs["l_data"] = l_data.data
            logs["l_semantic"] = l_semantic.data
            logs["l1_penalty"] = l1_penalty.data
            logs["satisfaction"] = avg_satisfaction.data

            if val_facts is not None:
                logs["val_accuracy"] = self.evaluate_accuracy(val_facts)

            if (epoch + 1) % 10 == 0 or epoch == self.epochs - 1:
                val_msg = (
                    f", Acc. Validação: {logs['val_accuracy']:.4f}"
                    if val_facts is not None
                    else ""
                )
                self.logger.info(
                    f"Época [{epoch+1}/{self.epochs}], Perda: {logs['loss']:.4f}, "
                    f"Satisfação das Regras: {logs['satisfaction']:.4f}{val_msg}"
                )

            self.callbacks_handler("on_epoch_end", epoch, logs)

        self.callbacks_handler("on_train_end", logs)
        self.logger.info("--- ✅ Treinamento Concluído ---")

    def callbacks_handler(self, method_name: str, *args):
        for cb in self.callbacks:
            method = getattr(cb, method_name)
            method(*args)
