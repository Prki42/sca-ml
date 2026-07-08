# %% [markdown]
# ## Modeli, evaluacija i traženje hiperparametara
#
# Implementirane su dve klase arhitektura - potpuno povezane neuronske mreže (MLP) i konvolutivne mreže (CNN). Namera je bila da se arhitekture što većim delom parametrizuju i optimalne vrednosti hiperparametara traže pomoću [Optune](https://optuna.org/).
#
# Optuni kao metriku za poređenje modela stavljamo površinu ispod rang krive a samu rang krivu računamo za unapred određen broj trace-eva iz attack skupa.
#
# U `trial_config/` direktorijumu se nalaze konfiguracioni `yaml` fajlovi koji definišu dodatne parametre pretrage koje se tiču evaluacije i treninga poput broja epoha, broja merenja koji koristimo pri određivanju rang krive i druge.
#
# TODO: svašta
