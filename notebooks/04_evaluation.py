# %% [markdown]
# ### Evaluacija modela
#
# U kontekstu side channel kriptoanaliza - modeli se najčešće porede po njihovim rang krivama.

# %% jupyter={"source_hidden": true}
import sys

sys.path.insert(0, "..")
# %load_ext autoreload
# %autoreload 2

# %% jupyter={"source_hidden": true}
# %matplotlib inline
import matplotlib.pyplot as plt

# %% jupyter={"source_hidden": true}
from datasets import ASCADv1FixedKey
from models import mlp, cnn
from train import train_model, split_profiling, load_trained_model, TrainResult
from evaluate import compute_rank_curve


# %% jupyter={"source_hidden": true}
def plot_rank_curves(curves):
    alpha = 1.0
    if len(curves) > 1:
        alpha = 0.55
    fig, ax = plt.subplots(figsize=(12,4))
    for c in curves:
        curve = c["data"]
        ax.plot(range(1, len(curve) + 1), curve, label=c["name"], alpha=alpha)
    ax.set_xlabel("Broj attack trace-ova")
    ax.set_ylabel("Rang")
    if len(curves) > 1:
        ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return ax

def plot_rank_curve(curve, name=None):
    return plot_rank_curves([{"data": curve, "name": name}])

def plot_losses(result: TrainResult, save_path: str | None = None):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(result.train_losses, label="train")
    ax.plot(result.val_losses, label="val")
    ax.axvline(x=result.best_epoch, color="red", linestyle="--", label="best")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return ax


# %% [markdown]
# Učitavanje profile (trening/validacija) i attack skupova:

# %%
ascad = ASCADv1FixedKey("../data/ASCAD/ATMEGA_AES_v1_fixed_key/ASCAD.h5")
profile_traces, profile_labels, profile_plaintexts = ascad.get_profiling()
train_traces, train_labels, val_traces, val_labels, val_plaintexts = split_profiling(
    profile_traces, profile_labels, profile_plaintexts
)
attack_tr, attack_pt, key = ascad.get_attack()

# %% [markdown]
# #### MLP
# Parametri za treniranje i hiperparametri MLP modela dobijnog korišćenjem Optune:
# ```
# {
#     "n_layers": 3,
#     "width": 100,
#     "dropout": 0.3370096425569424,
#     "activation": "leaky_relu",
#     "lr": 0.00016529586383204518,
#     "batch_size": 128,
# }
# ```
# Ispod se može naći kod za treniranje i evaluaciju tog modela:

# %%
# Inicijalizacija i trening
mlp_model = mlp.build_mlp(
    ascad.input_dim,
    ascad.num_classes,
    n_layers=3,
    width=100,
    dropout=0.3370096425569424,
    activation="leaky_relu",
)
mlp_trening = train_model(
    mlp_model,
    train_traces,
    train_labels,
    val_traces,
    val_labels,
    epochs=200,
    batch_size=128,
    lr=0.00016529586383204518,
    patience=100,
)
mlp_model = mlp_trening.model

# %%
# Alternativno, učitaj iz fajla
mlp_model, mlp_data = load_trained_model("../results/models/mlp_best.pt", mlp.build_mlp)
mlp_trening = TrainResult(model = mlp_model, train_losses = mlp_data["train_losses"], val_losses= mlp_data["val_losses"], best_epoch = mlp_data["best_epoch"])

# %%
plot_losses(mlp_trening)
plt.show()

# %%
# Evaluacija
mlp_curve = compute_rank_curve(mlp_model, attack_tr, attack_pt, key, ascad.leakage, 300)
plot_rank_curve(mlp_curve)
plt.show()

# %% [markdown]
# #### CNN
# Parametri za treniranje i hiperparametri CNN modela dobijenog korišćenjem Optune:
# ```
# {
#     "n_conv_layers": 1,
#     "n_filters": 64,
#     "kernel_size": 5,
#     "pool_size": 4,
#     "n_fc_layers": 3,
#     "fc_width": 100,
#     "dropout": 0.18496080246135194,
#     "activation": "selu",
#     "pool_type": "avg",
#     "conv_batchnorm": False,
#     "lr": 0.0003500608182408922,
#     "batch_size": 256,
# }
# ```
# Ispod se može naći kod za treniranje i evaluaciju tog modela:

# %%
# Inicijalizacija i trening:
cnn_model = cnn.build_cnn(
    ascad.input_dim,
    ascad.num_classes,
    n_conv=1,
    n_filters=64,
    kernel_size=5,
    pool_size=4,
    n_fc_layers=3,
    fc_width=100,
    dropout=0.18496080246135194,
    activation='selu',
    pool_type="avg",
    conv_batchnorm=False,
)

cnn_model = train_model(
    cnn_model,
    train_traces,
    train_labels,
    val_traces,
    val_labels,
    epochs=200,
    batch_size=256,
    lr=0.0003500608182408922,
    patience=70,
).model

# %%
# Alternativno, učitaj iz fajla
cnn_model, cnn_data = load_trained_model("../results/models/cnn_best.pt", cnn.build_cnn)
cnn_trening = TrainResult(model = cnn_model, train_losses = cnn_data["train_losses"], val_losses= cnn_data["val_losses"], best_epoch = cnn_data["best_epoch"])

# %%
plot_losses(cnn_trening)
plt.show()

# %%
# Evaluacija
cnn_curve = compute_rank_curve(cnn_model, attack_tr, attack_pt, key, ascad.leakage, 300)
plot_rank_curve(cnn_curve)
plt.show()

# %% [markdown]
# #### Poređenje

# %%
plot_rank_curves(
    [
        {"data": mlp_curve, "name": "MLP"},
        {"data": cnn_curve, "name": "CNN"},
    ]
)
plt.show()

# %% [markdown]
# Dobijeni modeli su bolji nego modeli u originalnom ASCAD radu.

# %% [markdown]
# TODO: za svrhe prezentacije dodati na grafik i modele iz pomenutog ASCAD rada
#
# TODO: za svrhe prezentacije dodati i ponasanje modela pri dodatoj desinhronizaciji
