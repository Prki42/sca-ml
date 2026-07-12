from collections.abc import Callable

import optuna
import torch.nn as nn

from .cnn import cnn_ascadv1_desync50_from_trial, cnn_from_trial
from .mlp import mlp_ascadv1_desync50_from_trial, mlp_from_trial

# registry: maps config name -> trial builder function
TRIAL_MODELS: dict[str, Callable[[optuna.Trial, int, int], nn.Module]] = {
    "mlp": mlp_from_trial,
    "cnn": cnn_from_trial,
    "cnn_ascadv1_desync50": cnn_ascadv1_desync50_from_trial,
    "mlp_ascadv1_desync50": mlp_ascadv1_desync50_from_trial,
}
