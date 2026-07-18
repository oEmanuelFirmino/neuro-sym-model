from .optimizer import SGD, AdamW
from .metrics import time_to_generalization, post_threshold_dip_count

__version__ = "0.1.0"
__author__ = "Emanuel Firmino"

__all__ = [
    "SGD",
    "AdamW",
    "time_to_generalization",
    "post_threshold_dip_count",
]
