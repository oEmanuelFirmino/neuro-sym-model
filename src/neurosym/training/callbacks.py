from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Dict
import logging

if TYPE_CHECKING:
    from .trainer import Trainer

from src.neurosym.training.saver import save_model

logger = logging.getLogger("CallbackSystem")


@abstractmethod
class Callback:
    def set_trainer(self, trainer: "Trainer"):
        self.trainer = trainer
        self.logs: Dict = {}

    def on_train_begin(self, logs: Dict = None):
        self.logs = logs or {}
        logger.debug(f"Treino iniciado. Logs: {self.logs}")

    def on_train_end(self, logs: Dict = None):
        self.logs = logs or {}
        logger.debug(f"Treino finalizado. Logs: {self.logs}")

    def on_epoch_begin(self, epoch: int, logs: Dict = None):
        self.logs = logs or {}
        logger.debug(f"Início do epoch {epoch}. Logs: {self.logs}")

    def on_epoch_end(self, epoch: int, logs: Dict = None):
        self.logs = logs or {}
        logger.debug(f"Fim do epoch {epoch}. Logs: {self.logs}")


class ModelCheckpoint(Callback):
    def __init__(
        self,
        filepath: str,
        monitor: str = "loss",
        mode: str = "min",
        save_best_only: bool = True,
    ):
        super().__init__()
        self.filepath = Path(filepath)
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode

        self.best = float("inf") if mode == "min" else float("-inf")
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

    def on_train_begin(self, logs: Dict = None):
        super().on_train_begin(logs)
        logger.info(
            f"ModelCheckpoint iniciado. Monitorando '{self.monitor}' "
            f"com modo='{self.mode}', save_best_only={self.save_best_only}."
        )
        self.best = float("inf") if self.mode == "min" else float("-inf")

    def on_train_end(self, logs: Dict = None):
        super().on_train_end(logs)
        logger.info(
            f"ModelCheckpoint finalizado. Melhor valor registrado para "
            f"'{self.monitor}' foi {self.best:.4f}."
        )

    def on_epoch_begin(self, epoch: int, logs: Dict = None):
        super().on_epoch_begin(epoch, logs)
        logger.debug(
            f"ModelCheckpoint: início do epoch {epoch}, melhor valor atual = {self.best:.4f}"
        )

    def on_epoch_end(self, epoch: int, logs: Dict = None):
        super().on_epoch_end(epoch, logs)

        current_val = self.logs.get(self.monitor)
        if current_val is None:
            logger.warning(
                f"ModelCheckpoint: métrica '{self.monitor}' não encontrada nos logs: {self.logs}"
            )
            return

        improved = (
            self.mode == "min"
            and current_val < self.best
            or self.mode == "max"
            and current_val > self.best
        )

        if self.save_best_only and not improved:
            return

        if improved:
            logger.info(
                f"\nModelCheckpoint: {self.monitor} melhorou "
                f"de {self.best:.4f} para {current_val:.4f}. "
                f"Salvando modelo em {self.filepath}"
            )
            self.best = current_val
        else:
            logger.info(
                f"\nModelCheckpoint: salvando modelo do epoch {epoch} "
                f"em {self.filepath} (save_best_only=False)"
            )

        save_model(
            self.filepath,
            self.trainer.interpreter.predicate_map,
            self.trainer.interpreter.grounding_env,
        )
