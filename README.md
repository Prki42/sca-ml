# Mašinsko učenje 25/26 projekat

Tema: **Side-channel kriptoanaliza**

Članovi tima:
- Maksim Bojović 64/2022
- Tijana Mijalkov 60/2022
- Aleksa Prtenjača 52/2022

## Tehnikalije

Podrazumeva se da je alat `uv` dostupan. Ukoliko `uv` nije prisutan - sve instrukcije se lako mogu prevesti na `pip` ekvivalentne komande.

U zavisnosti od toga da li želite da `torch` izvršavate na procesoru ili CUDA grafici izvršiti jednu od datih komandi. **Napomena** - ove instrukcije interno pozivaju `uv` - ukoliko `uv` nije dostupan prevesti komande iz `Makefile`-a u odgovarajuće `pip` komande.
```sh
make setup-cpu
make setup-gpu
```

Ukoliko je `jupyter-lab` globalno instaliran, da bi se trenutno virtuelno okruženje registrovalo kao kernel pokrenuti:
```sh
uv run python -m ipykernel install --user --name sca-ml
```
inače, instalirati `jupyter-lab` u trenutnom `.venv`-u koristeći:
```sh
uv pip install jupyterlab
```

Sveske u `notebooks/` direktorijumu su u `py:percent` formatu radi (smislenog) verzionisanja. Od njih se mogu generisati uparene `.ipynb` sveske i koristiti normalno. Pogledati `jupytext` [dokumentaciju](https://jupytext.org/getting-started/introduction/) za više detalja i instalaciju ekstenzije.

## O projektu

Za sada pogledati sadržaj `notebooks/` direktorijuma.

Okvirna organizacija:
- `data/` - sadrži dataset-ove
- `datasets/` - bazna klasa za opisivanje operacija nad datasetom i konkretna implementacija za ASCADv1 bazu
- `models/` - implementacija arhitektura
- `trial_configs/` - opis Optuninih trial-a za traženje hiperparametara
- `analysis.py` - utility strukture i metodi za ("efikasno") računanje SNR-a
- `train.py` - funkcije za trening/validacija split i samo treniranje
- `evaluate.py` - računanje rang krive za istrenirani model
- `search.py` - pokretanje Optunine pretrage za datu konfiguraciju iz `trial_configs/`

## Dataset-ovi

Pogledati `README.md` u `data/` direktorijumu.

## Reference

- [Study of Deep Learning Techniques for Side-Channel Analysis and Introduction to ASCAD Database](https://eprint.iacr.org/2018/053)
- [SoK: Deep Learning-based Physical Side-channel Analysis](https://dl.acm.org/doi/full/10.1145/3569577)

