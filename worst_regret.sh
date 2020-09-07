for i in {1..10}; do
  python mnl_bandit.py --data --fig --random_params \
  --horizon=1000000 --freq=10000 --card_limit=4 --trials=20 --processes=20 \
  --params_filename="params_${i}.json" --output_filename="data_${i}.out" --figure_filename="fig_${i}.pdf"
done
