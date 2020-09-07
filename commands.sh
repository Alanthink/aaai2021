# simple run of synthetic data
python mnl_bandit.py --data --fig --random_params \
--horizon=10000 --freq=100 --card_limit=4 --trials=10 --processes=10 --product_num=10

# real data
python mnl_bandit.py --cvar_data --cvar_fig \
--horizon=1000000 --freq=1000 --card_limit=100 --trials=40 --processes=40 \
--random_neighbors=10 --percentile=5
