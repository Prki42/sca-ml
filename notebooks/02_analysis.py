# %% [markdown]
# ### Nalaženje relevantnog vremenskog intervala - Signal-to-noise ratio
# U našem konkretnom projektu i sa našom konkretnom bazom, cilj je da se nađe vrednost _trećeg bajta_ ključa (ključ se inače sastoji od 16 bajtova). Originalna baza, iz koje je naša izvedena, se sastoji iz čak 100.000 trenutaka merenja. Dakle, bilo je potrebno suziti tu bazu na samo one trenutke (tj onaj vremenski interval) u okviru kog najviše figuriše ciljani bajt ključa. Ovde se javlja jasan problem u nalaženju tog vremenskog intervala. Za to ćemo se poslužiti merom SNR (odnos signala i šuma, engl. _signal-to-noise ratio_).
#
# Konkretno, mi imamo neka merenja potrošnje struje $\{T(t)|t\in \{1,...,N\}\}$, kao i "osetljivu"\, promenljivu $Z$ (u našem slučaju _Sbox_ funkcija). SNR se definiše kao: $$SNR(t) = \frac{D_Z(E(T(t)|Z))}{E_Z(D(T(t)|Z))}$$
# gde se odgovarajuće očekivanje i disperzija računaju po razbijanju $Z=Z_0 \sqcup Z_1\sqcup ...\sqcup Z_{255}$, gde je $Z_i$ skup opservacija za koje _Sbox_ (odnosno proizvoljna funkcija) uzima vrednost $i$.

# %% jupyter={"source_hidden": true}
import sys
sys.path.insert(0, "..")
# %load_ext autoreload
# %autoreload 2

# %% jupyter={"source_hidden": true}
# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

# %% jupyter={"source_hidden": true}
import analysis
from datasets.ascad_v1 import ASCADv1FixedKey, AES_SBOX, compute_snr_ascad

# %%
# Precomputed vrednosti za SNR dijagrame

snrs = {
    "snr2": np.load("../results/snr2.npy"),
    "snr4": np.load("../results/snr4.npy")
}

# %%
# Racunanje SNR za izvnornu ASCAD bazu koja zauzima ~7GB moze potrajati koj minut

target_byte = 2
snrs = compute_snr_ascad(
    "../data/ASCAD/ATMEGA_AES_v1_fixed_key/ATMega8515_raw_traces.h5",
    {
        "snr2": lambda meta: (
            AES_SBOX[
                meta["plaintext"][:, target_byte] ^ meta["key"][:, target_byte]
            ]
            ^ meta["masks"][:, -1] # r_out
        ),
        "snr4": lambda meta: (
            AES_SBOX[
                meta["plaintext"][:, target_byte] ^ meta["key"][:, target_byte]
            ]
            ^ meta["masks"][:, target_byte - 2] # r[target_byte]
        ),
    },
)

# %%
ax = analysis.plot_snr(snrs, 45400, 48000, None)
plt.show()

# %% [markdown]
# Na osnovu SNR analize možemo zaključiti koji vremenski interval je relevantan za vrednost koju hoćemo da izvučemo. Ovim znatno smanjujemo dimenziju podataka i dobijamo dataset koji možemo dalje koristiti direktno kao ulaz u klasifikatore. Autori ASCAD baze su već suzili originalna merenja iz `ATMega8515_raw_traces.h5` dataset-a na manji dataset koji sadrži segment relevantan za napadanje trećeg bajta ključa i taj dataset je `ASCAD.h5`.
