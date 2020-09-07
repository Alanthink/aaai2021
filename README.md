# Steps to generate figures

## Preliminary

Install package `https://github.com/Alanthink/banditpylib`.

## Figure 1

* Run `worst_regret.h` to generate 10 random instances and their trials
* Move generated files to `arxiv` folder (currently `arxiv` contains all files generated last time of running)
* Run `python mnl_bandit.py --final` to generate Figure 1

## Figure 2

Real parameters are already manipulated and stored in file `real_params.json`. See file `car_data_processing.ipynb` on the code to manipulate data. Just run 
```
python mnl_bandit.py --cvar_data --cvar_fig \
--horizon=1000000 --freq=1000 --card_limit=100 --trials=40 --processes=40 \
--random_neighbors=10 --percentile=5
```
to generate Figure 2.
