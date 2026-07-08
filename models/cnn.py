import optuna
import torch
import torch.nn as nn


class SCA_CNN(nn.Module):
    def __init__(self, conv_layers: nn.Sequential, fc_layers: nn.Sequential):
        super().__init__()
        self.conv = conv_layers
        self.fc = fc_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # (B, 1, D)
        x = self.conv(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


def build_cnn(
    input_dim: int,
    num_classes: int,
    n_conv: int,
    n_filters: int,
    kernel_size: int,
    pool_size: int,
    n_fc: int,
    fc_width: int,
    dropout: float,
    activation: str = "relu",
    pool_type: str = "avg",
    conv_batchnorm: bool = True,
):
    act_fn = {
        "relu": nn.ReLU,
        "selu": nn.SELU,
        "leaky_relu": nn.LeakyReLU,
    }[activation]

    pool_fn = {"avg": nn.AvgPool1d, "max": nn.MaxPool1d}[pool_type]

    conv_layers: list[nn.Module] = []
    in_channels = 1  # single-channel for 1D signal
    length = input_dim
    for i in range(n_conv):
        out_channels = n_filters * (2**i)
        padding = kernel_size // 2
        conv_layers.append(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        )
        if conv_batchnorm:
            conv_layers.append(nn.BatchNorm1d(out_channels))
        conv_layers.append(act_fn())
        conv_layers.append(pool_fn(pool_size))
        length = length // pool_size
        in_channels = out_channels

    flat_dim = in_channels * length

    fc_layers: list[nn.Module] = []
    for i in range(n_fc):
        in_features = flat_dim if i == 0 else fc_width
        fc_layers.append(nn.Linear(in_features, fc_width))
        fc_layers.append(act_fn())
        if dropout > 0:
            fc_layers.append(nn.Dropout(dropout))
    fc_layers.append(nn.Linear(fc_width, num_classes))

    return SCA_CNN(nn.Sequential(*conv_layers), nn.Sequential(*fc_layers))


def cnn_from_trial(trial: optuna.Trial, input_dim: int, num_classes: int) -> nn.Module:
    """Build CNN with Optuna-sampled hyperparameters."""
    return build_cnn(
        input_dim,
        num_classes,
        n_conv=trial.suggest_int("n_conv_layers", 1, 4),
        n_filters=trial.suggest_categorical("n_filters", [8, 16, 32, 64]),
        kernel_size=trial.suggest_categorical("kernel_size", [3, 5, 11, 21]),
        pool_size=trial.suggest_categorical("pool_size", [2, 4]),
        n_fc=trial.suggest_int("n_fc_layers", 1, 3),
        fc_width=trial.suggest_categorical("fc_width", [100, 200, 400]),
        dropout=trial.suggest_float("dropout", 0.0, 0.5),
        activation=trial.suggest_categorical(
            "activation", ["relu", "selu", "tanh", "leaky_relu"]
        ),
        pool_type=trial.suggest_categorical("pool_type", ["avg", "max"]),
        conv_batchnorm=trial.suggest_categorical("conv_batchnorm", [True, False]),
    )
