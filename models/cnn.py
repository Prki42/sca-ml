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


def cnn_ascadv1_desync50_from_trial(
    trial: optuna.Trial, input_dim: int, num_classes: int
) -> nn.Module:

    return build_cnn_ascadv1_desync50(
        input_dim,
        num_classes,
        n_filters=trial.suggest_categorical("n_filters", [4, 8, 16, 32]),
        last_pool_type=trial.suggest_categorical("last_pool_type", ["avg", "max"]),
        # before last conv layer length is 14 so this gives us final length in [2, 7]
        last_pool_size=trial.suggest_int("last_pool_size", low=2, high=5),
        last_kernel_size=trial.suggest_categorical(
            "last_kernel_size", [1, 3, 5, 7, 11]
        ),
        n_fc_layers=trial.suggest_int("n_fc_layers", 1, 5),
        # fc input dimension is n_filters * 4 * (14 // last_pool_size) which is in [32, 896]
        fc_width=trial.suggest_categorical("fc_width", [5, 10, 15, 25, 50, 100]),
    )


def build_cnn_ascadv1_desync50(
    input_dim: int,
    num_classes: int,
    n_filters: int,
    last_pool_type: str,
    last_pool_size: int,
    last_kernel_size: int,
    n_fc_layers: int,
    fc_width: int,
):
    conv_layers: list[nn.Module] = []

    last_pool_fn = {"avg": nn.AvgPool1d, "max": nn.MaxPool1d}[last_pool_type]

    length = input_dim

    # Reduce dimensionality
    conv_layers.append(nn.Conv1d(1, n_filters, 1, padding=0))
    conv_layers.append(nn.SELU())
    conv_layers.append(nn.AvgPool1d(2, 2))
    length //= 2

    # Desync
    conv_layers.append(nn.Conv1d(n_filters, n_filters * 2, 25, padding=12))
    conv_layers.append(nn.SELU())
    conv_layers.append(nn.AvgPool1d(25, 25))
    length //= 25

    # Another one
    conv_layers.append(
        nn.Conv1d(
            n_filters * 2,
            n_filters * 4,
            last_kernel_size,
            padding=last_kernel_size // 2,
        )
    )
    conv_layers.append(nn.SELU())
    conv_layers.append(last_pool_fn(last_pool_size, last_pool_size))
    length //= last_pool_size

    flat_dim = n_filters * 4 * length

    fc_layers: list[nn.Module] = []
    for i in range(n_fc_layers):
        in_features = flat_dim if i == 0 else fc_width
        fc_layers.append(nn.Linear(in_features, fc_width))
        fc_layers.append(nn.SELU())
    fc_layers.append(nn.Linear(fc_width, num_classes))

    # LeCun init for selu in conv layers
    for m in conv_layers:
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="linear")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    # LeCun init for selu in linear layers
    for m in fc_layers:
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="linear")
            nn.init.zeros_(m.bias)

    return SCA_CNN(nn.Sequential(*conv_layers), nn.Sequential(*fc_layers))


def build_cnn(
    input_dim: int,
    num_classes: int,
    n_conv: int,
    n_filters: int,
    kernel_size: int,
    pool_size: int,
    n_fc_layers: int,
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
    for i in range(n_fc_layers):
        in_features = flat_dim if i == 0 else fc_width
        fc_layers.append(nn.Linear(in_features, fc_width))
        fc_layers.append(act_fn())
        if dropout > 0:
            fc_layers.append(nn.Dropout(dropout))
    fc_layers.append(nn.Linear(fc_width, num_classes))

    # LeCun init
    if activation == "selu":
        for m in [*conv_layers, *fc_layers]:
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    return SCA_CNN(nn.Sequential(*conv_layers), nn.Sequential(*fc_layers))


def cnn_from_trial(trial: optuna.Trial, input_dim: int, num_classes: int) -> nn.Module:
    """Build CNN with Optuna-sampled hyperparameters."""

    n_conv = trial.suggest_int("n_conv_layers", 1, 4)
    pool_size = trial.suggest_categorical("pool_size", [2, 4])
    kernel_size = trial.suggest_categorical("kernel_size", [3, 5, 11, 21])
    min_dim = input_dim // (pool_size**n_conv)

    if min_dim < 1 or min_dim < kernel_size:
        raise optuna.TrialPruned()

    activation = trial.suggest_categorical("activation", ["relu", "selu", "leaky_relu"])
    conv_batchnorm = trial.suggest_categorical("conv_batchnorm", [True, False])

    if activation == "selu" and conv_batchnorm:
        raise optuna.TrialPruned()

    return build_cnn(
        input_dim,
        num_classes,
        n_conv=n_conv,
        n_filters=trial.suggest_categorical("n_filters", [8, 16, 32, 64]),
        kernel_size=kernel_size,
        pool_size=pool_size,
        n_fc_layers=trial.suggest_int("n_fc_layers", 1, 3),
        fc_width=trial.suggest_categorical("fc_width", [100, 200, 400]),
        dropout=trial.suggest_float("dropout", 0.0, 0.5),
        activation=activation,
        pool_type=trial.suggest_categorical("pool_type", ["avg", "max"]),
        conv_batchnorm=conv_batchnorm,
    )
