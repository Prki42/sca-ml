from collections.abc import Callable

import optuna
import torch.nn as nn

from .mlp import mlp_from_trial
from .cnn import cnn_from_trial

TRIAL_MODELS: dict[str, Callable[[optuna.Trial, int, int], nn.Module]] = {
    "mlp": mlp_from_trial,
    "cnn": cnn_from_trial
}
