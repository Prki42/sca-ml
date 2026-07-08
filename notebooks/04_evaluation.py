# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: sca-ml
#     language: python
#     name: sca-ml
# ---

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
from train import train_model, split_profiling
from evaluate import compute_rank_curve


# %% jupyter={"source_hidden": true}
def plot_curves(curves):
    alpha = 1.0
    if len(curves) > 1:
        alpha = 0.55
    for c in curves:
        curve = c["data"]
        plt.plot(range(1, len(curve) + 1), curve, label = c["name"], alpha=alpha)
    plt.xlabel("Broj attack trace-ova")
    plt.ylabel("Rang")
    if (len(curves) > 1):
        plt.legend()
    # plt.title("")
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_curve(curve, name = None):
    plot_curves([{"data": curve, "name": name}])


# %% [markdown]
# Učitavanje profile (trening/validacija) i attack skupova:

# %%
ascad = ASCADv1FixedKey("../data/ASCAD/ATMEGA_AES_v1_fixed_key/ASCAD.h5")
profile_traces, profile_labels, profile_plaintexts = ascad.get_profiling()
train_traces, train_labels, val_traces, val_labels, val_plaintexts = (
    split_profiling(profile_traces, profile_labels, profile_plaintexts)
)
attack_tr, attack_pt, key = ascad.get_attack()
attack_tr, attack_pt, key = ascad.get_attack()

# %% [markdown]
# #### MLP
# Parametri za treniranje i hiperparametri MLP modela dobijnog korišćenjem Optune:
# ```
# {'n_layers': 3, 'width': 600, 'dropout': 0.47187931041403586, 'activation': 'leaky_relu', 'lr': 8.830185072117689e-05, 'batch_size': 128}
# ```
# Ispod se može naći kod za treniranje i evaluaciju tog modela:

# %%
# Inicijalizacija i trening
mlp_model = mlp.build_mlp(ascad.input_dim, ascad.num_classes, n_layers=3, width=600, dropout = 0.47187931041403586, activation="leaky_relu")
mlp_model = train_model(model, train_traces, train_labels, val_traces, val_labels, epochs=200, batch_size=128, lr=8.830185072117689e-05, patience = 100)

# %%
# Alternativno, učitaj iz fajla
# TODO

# %%
# Evaluacija
mlp_curve = compute_rank_curve(mlp_model, attack_tr, attack_pt, key, ascad.leakage, 300)
plot_curve(mlp_curve)

# %% [markdown]
# #### CNN
# Parametri za treniranje i hiperparametri CNN modela dobijenog korišćenjem Optune:
# ```
# {'n_conv_layers': 1, 'n_filters': 32, 'kernel_size': 3, 'pool_size': 2, 'n_fc_layers': 3, 'fc_width': 200, 'dropout': 0.4191105132000773, 'lr': 1.344787118591703e-05, 'batch_size': 256}
# ```
# Ispod se može naći kod za treniranje i evaluaciju tog modela:

# %%
# Inicijalizacija i trening:
cnn_model = cnn.build_cnn(ascad.input_dim, ascad.num_classes,
        n_conv = 1,
        n_filters = 32,
        kernel_size = 3,
        pool_size = 2,
        n_fc = 3,
        fc_width = 200,
        dropout = 0.4191105132000773
)
cnn_model = train_model(model, train_traces, train_labels, val_traces, val_labels, epochs=400, batch_size=256, lr=1.344787118591703e-05, patience = 70)

# %%
# Alternativno, učitaj iz fajla
# TODO

# %%
# Evaluacija
cnn_curve = compute_rank_curve(cnn_model, attack_tr, attack_pt, key, ascad.leakage, 300)
plot_curve(cnn_curve)

# %% [markdown]
# #### Poređenje

# %%
plot_curves(
[
    {
        "data": mlp_curve,
        "name": "MLP"
    },
    {
        "data": cnn_curve,
        "name": "CNN"
    },
])
