# %% [markdown]
# ## Modeli i traženje hiperparametara
#
# Implementirane su dve klase arhitektura - potpuno povezane neuronske mreže (MLP) i konvolutivne mreže (CNN). Namera je bila da se arhitekture što većim delom parametrizuju i optimalne vrednosti hiperparametara traže pomoću [Optune](https://optuna.org/).
#
# Optuni kao metriku za poređenje modela stavljamo površinu ispod rang krive a samu rang krivu računamo za unapred određen broj trace-eva iz validacionog skupa.
#
# U `trial_configs/` direktorijumu se nalaze konfiguracioni `yaml` fajlovi koji definišu dodatne parametre pretrage koje se tiču evaluacije i treninga poput broja epoha, broja merenja koji koristimo pri određivanju rang krive i druge.
#
# U `results/models/` se nalaze neki istrenirani modeli (`.pt` fajlovi) zajedno sa svim parametrima arhitekture i treninga, kao i podacima o treniranju (train/val losses i best epoch).

# %% [markdown]
# ### MLP
#
# Klasifikacioni MLP model sa $n-1$ skrivenih layer-a definišemo kao $ s \circ [ \lambda ]^n $ gde je $s$ softmax funkcija, $\lambda : \mathbb{R}^{d_1} \rightarrow \mathbb{R}^{d_2} : x \mapsto \sigma(Ax + b)$ linearno preslikavanje komponovano sa aktivacionom funkcijom $\sigma$ (npr. ReLU) primenjena član po član. Svi skriveni layer-i su iste dimenzije $d$. Samim tim jedini hiper-parametri modela koje imamo su $n$, $d$ i izbor aktivacione funkcije $\sigma$.

# %% [markdown]
# ### CNN
#
# Klasifikacioni CNN model u opštem slučaju definišemo kao $ s \circ [\lambda]^{n_1} \circ [ \ \delta \circ [ \sigma \circ \gamma ]^{n_2} \ ]^{n_3} $ gde je $s$ softmax funkcija, $\gamma$ konvolucijski sloj, $\sigma$ aktivaciona funkcija i $\delta$ pooling sloj.
#
# Koristimo domensko znanje zarad fiksiranja određenih aspekata CNN arhitekture. Ono što se u literaturi pokazalo kao plodan pristup jeste sledeće:
# - Jedan konvolucijski blok sa filterima dimenzije $1$ i AvgPool-om sa stride-om od $2$. Svrha ovog bloka jeste samo da se umanji dimenzionalnost ulaza.
# - Jedan konvolucijski blok sa filterima dimenzije $S/2$ i AvgPool-om sa stride-om isto $S/2$ gde je $S$ količina desinhronizacije.
#
# Na to dodajemo i još jedan konvolucijski blok koji dajemo Optuni da traži.
