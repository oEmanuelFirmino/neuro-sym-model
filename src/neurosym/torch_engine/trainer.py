"""Equivalente batched de `src/neurosym/training/trainer.py`: mesma composição de
loss (L_total = L_data + lambda*L_semantic + gamma*||W||_1), mas o loop
"um `eval_formula` por fato/regra por época" vira "poucas chamadas em lote" via
`compile_formulas`/`BatchedInterpreter.eval_group`.

Contrato de device: `BatchedTrainer` NÃO move o `model` para `device` sozinho --
o chamador deve montar `model`, chamar `model.to(device)` e SÓ DEPOIS construir o
`optimizer` sobre `model.parameters()` (ordem padrão do PyTorch; mover o modelo
depois de criar o optimizer pode dessincronizar as referências de parâmetro que o
optimizer rastreia). `BatchedTrainer` só usa `device` para colocar os tensores de
`leaf_idx`/`targets` compilados no mesmo lugar que os parâmetros do modelo.
"""

import logging
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from src.neurosym.logic import Formula
from src.neurosym.training.callbacks import Callback

from .compile import CompiledGroup, compile_formulas
from .interpreter import BatchedInterpreter
from .model import DLGModel
from .modules import l1_penalty

Fact = Tuple[Formula, float]


class BatchedTrainer:
    def __init__(
        self,
        model: DLGModel,
        interpreter: BatchedInterpreter,
        optimizer: torch.optim.Optimizer,
        epochs: int,
        device: str = "cpu",
        callbacks: Optional[List[Callback]] = None,
        lambda_semantic: float = 1.0,
        gamma_l1: float = 0.0,
        val_threshold: float = 0.5,
        accuracy_fn: Optional[Callable[[List[Fact]], Optional[float]]] = None,
        val_eval_every: int = 1,
    ):
        self.model = model
        self.interpreter = interpreter
        self.optimizer = optimizer
        self.epochs = epochs
        self.device = torch.device(device)
        self.callbacks = callbacks if callbacks else []
        self.lambda_semantic = lambda_semantic
        self.gamma_l1 = gamma_l1
        self.val_threshold = val_threshold
        self.accuracy_fn = accuracy_fn
        self.val_eval_every = max(1, val_eval_every)
        self._last_val_accuracy: Optional[float] = None
        self.logger = self._setup_logger()

        for cb in self.callbacks:
            cb.set_trainer(self)

    def _setup_logger(self):
        logger = logging.getLogger("BatchedNeuroSymbolicTrainer")
        logger.setLevel(logging.INFO)
        if not logger.hasHandlers():
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                "%(asctime)s | %(name)s | %(message)s", datefmt="%H:%M:%S"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def _compile_facts(self, facts: List[Fact]) -> List[CompiledGroup]:
        if not facts:
            return []
        vocab = self.model.grounding_env.vocab
        groups = compile_formulas(
            [f for f, _ in facts], vocab, targets=[t for _, t in facts]
        )
        return [
            CompiledGroup(
                g.signature,
                g.template,
                g.leaf_idx.to(self.device),
                g.targets.to(self.device) if g.targets is not None else None,
            )
            for g in groups
        ]

    def _compile_rules(self, rules: List[Formula]) -> List[CompiledGroup]:
        if not rules:
            return []
        vocab = self.model.grounding_env.vocab
        groups = compile_formulas(rules, vocab)
        return [
            CompiledGroup(g.signature, g.template, g.leaf_idx.to(self.device), None)
            for g in groups
        ]

    def _evaluate_accuracy_from_groups(
        self, groups: List[CompiledGroup], threshold: float
    ) -> Optional[float]:
        correct, total = 0, 0
        with torch.no_grad():
            for g in groups:
                preds = self.interpreter.eval_group(g)
                correct += int(((preds >= threshold) == (g.targets >= threshold)).sum().item())
                total += preds.shape[0]
        return correct / total if total else None

    def evaluate_accuracy(
        self, facts: List[Fact], threshold: Optional[float] = None
    ) -> Optional[float]:
        if not facts:
            return None

        if self.accuracy_fn is not None:
            return self.accuracy_fn(facts)

        threshold = self.val_threshold if threshold is None else threshold
        return self._evaluate_accuracy_from_groups(self._compile_facts(facts), threshold)

    def fit(
        self,
        rules: List[Formula],
        facts: List[Fact],
        val_facts: Optional[List[Fact]] = None,
    ):
        self.logger.info("--- Iniciando Loop de Treinamento (batched) ---")

        fact_groups = self._compile_facts(facts)
        rule_groups = self._compile_rules(rules)
        # Pré-compilado uma vez (não a cada avaliação) quando accuracy_fn não
        # substitui a lógica padrão -- senão, com val_eval_every pequeno e um
        # conjunto de validação grande, recompilar a cada avaliação reintroduz o
        # overhead de laço Python que este motor existe para eliminar.
        val_groups = (
            self._compile_facts(val_facts)
            if val_facts is not None and self.accuracy_fn is None
            else None
        )

        logs: Dict[str, Any] = {}
        self.callbacks_handler("on_train_begin", logs)

        for epoch in range(self.epochs):
            self.callbacks_handler("on_epoch_begin", epoch, logs)
            self.optimizer.zero_grad()

            if fact_groups:
                sq_errors = [
                    (self.interpreter.eval_group(g) - g.targets) ** 2 for g in fact_groups
                ]
                l_data = torch.cat(sq_errors).mean()
            else:
                l_data = torch.tensor(0.0, device=self.device)

            if rule_groups:
                truths = [self.interpreter.eval_group(g) for g in rule_groups]
                avg_satisfaction = torch.cat(truths).mean()
                l_semantic = 1.0 - avg_satisfaction
            else:
                avg_satisfaction = torch.tensor(1.0, device=self.device)
                l_semantic = torch.tensor(0.0, device=self.device)

            l1 = l1_penalty(self.model.predicate_map)

            loss = l_data + self.lambda_semantic * l_semantic + self.gamma_l1 * l1
            loss.backward()
            self.optimizer.step()

            logs["loss"] = loss.item()
            logs["l_data"] = l_data.item()
            logs["l_semantic"] = l_semantic.item()
            logs["l1_penalty"] = l1.item()
            logs["satisfaction"] = avg_satisfaction.item()

            if val_facts is not None:
                is_last_epoch = epoch == self.epochs - 1
                if epoch % self.val_eval_every == 0 or is_last_epoch:
                    if val_groups is not None:
                        self._last_val_accuracy = self._evaluate_accuracy_from_groups(
                            val_groups, self.val_threshold
                        )
                    else:
                        self._last_val_accuracy = self.evaluate_accuracy(val_facts)
                logs["val_accuracy"] = self._last_val_accuracy

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
        self.logger.info("--- Treinamento Concluído ---")

    def callbacks_handler(self, method_name: str, *args):
        for cb in self.callbacks:
            method = getattr(cb, method_name)
            method(*args)
