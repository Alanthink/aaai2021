# Steps to generate figures

Note that all the following commands are run under the root directory i.e., `risk_aware_mnl`.

## Preliminary

Install package `https://github.com/Alanthink/banditpylib`.

## Figure 1

* Run `worst_regret.sh` to generate 10 random input instances and start simulations
* Move generated files in formats `data_*.out` or `params_*.json` to `arxiv` folder (currently `arxiv` contains all files generated from last time of running)
* Run `python mnl_bandit.py --final` to generate Figure 1

## Figure 2

Real parameters are already manipulated and stored in file `real_params.json`. See file `car_data_processing.ipynb` on the code to manipulate original data. Then just run 
```
python mnl_bandit.py --cvar_data --cvar_fig \
--horizon=1000000 --freq=1000 --card_limit=100 --trials=40 --processes=40 \
--random_neighbors=10 --percentile=5
```
to generate Figure 2.
