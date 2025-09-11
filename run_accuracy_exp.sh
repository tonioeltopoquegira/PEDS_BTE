#!/usr/bin/env zsh

# list of models you want to run
models=(
  #peds_fourier
  #mlpmod
  #peds_fourier_uq1
  mlp_ens_uq1
)

# number of trials per model (0 through 9)
for seed in {1..5}; do
  for model in $models; do
    echo "Running 1000_data $model seed=$seed"
    mpirun -n 4 python pipeline_peds.py 1000_data $model $seed
  done
done
