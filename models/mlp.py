import torch.nn as nn


def build_mlp(
    input_dim: int,
    num_classes: int,
    n_layers: int = 5,
    width: int = 200,
    activation: str = "relu",
) -> nn.Module:
    act_fn = {
        "relu": nn.ReLU,
        "selu": nn.SELU,
        "tanh": nn.Tanh,
        "leaky_relu": nn.LeakyReLU,
    }[activation]

    layers: list[nn.Module] = []
    for i in range(n_layers):
        in_features = input_dim if i == 0 else width
        layers.append(nn.Linear(in_features, width))
        layers.append(act_fn())
    layers.append(nn.Linear(width, num_classes))

    model = nn.Sequential(*layers)
    return model
