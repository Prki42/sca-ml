import optuna
import torch.nn as nn


def build_mlp(
    input_dim: int,
    num_classes: int,
    n_layers: int = 5,
    width: int = 200,
    dropout: float = 0.0,
    activation: str = "relu",
) -> nn.Module:
    act_fn = {
        "relu": nn.ReLU,
        "selu": nn.SELU,
        "leaky_relu": nn.LeakyReLU,
    }[activation]

    layers: list[nn.Module] = []
    for i in range(n_layers):
        in_features = input_dim if i == 0 else width
        layers.append(nn.Linear(in_features, width))
        layers.append(act_fn())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(width, num_classes))

    model = nn.Sequential(*layers)

    for m in model:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    return model


def mlp_from_trial(trial: optuna.Trial, input_dim: int, num_classes: int) -> nn.Module:
    """Build MLP with Optuna-sampled hyperparameters."""
    return build_mlp(
        input_dim,
        num_classes,
        n_layers=trial.suggest_int("n_layers", 2, 8),
        width=trial.suggest_categorical("width", [100, 200, 400, 600]),
        dropout=trial.suggest_float("dropout", 0.0, 0.5),
        activation=trial.suggest_categorical(
            "activation", ["relu", "selu", "tanh", "leaky_relu"]
        ),
    )
