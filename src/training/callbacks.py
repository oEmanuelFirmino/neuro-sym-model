import sys
from pathlib import Path
from typing import TYPE_CHECKING, Dict
import logging
import copy


if TYPE_CHECKING:
    from .trainer import Trainer


from src.training.saver import save_model
from src.interpreter import PredicateMap, GroundingEnv

logger = logging.getLogger("CallbackSystem")


class Callback:

    def set_trainer(self, trainer: "Trainer"):
        self.trainer = trainer

    def on_train_begin(self, logs: Dict = None):
        pass

    def on_train_end(self, logs: Dict = None):
        pass

    def on_epoch_begin(self, epoch: int, logs: Dict = None):
        pass

    def on_epoch_end(self, epoch: int, logs: Dict = None):
        pass


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
        self.best = float("inf") if mode == "min" else float("-inf")
        self.mode = mode

        self.filepath.parent.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, epoch: int, logs: Dict = None):
        current_val = logs.get(self.monitor)
        if current_val is None:
            return

        is_better = (self.mode == "min" and current_val < self.best) or (
            self.mode == "max" and current_val > self.best
        )

        if is_better:
            logger.info(
                f"\nModelCheckpoint: {self.monitor} melhorou de {self.best:.4f} para {current_val:.4f}. Salvando modelo em {self.filepath}"
            )
            self.best = current_val

            save_model(
                self.filepath,
                self.trainer.interpreter.predicate_map,
                self.trainer.interpreter.grounding_env,
            )
