#!/usr/bin/env zsh

# List of models to run
models=(
  peds_fourier
  mlpmod
  peds_fourier_uq1
  mlp_ens_uq1
)

# List of experiments from 100 to 2000 data
experiments=(
  #100_data
  #200_data
  #500_data
  2000_data
)

# Seeds to use (2 seeds per experiment-model pair)
seeds=(1 2)

# Run the experiments
for exp in $experiments; do
  for model in $models; do
    for seed in $seeds; do
      echo "Running $exp $model seed=$seed"
      mpirun -n 4 python pipeline_peds.py $exp $model $seed
    done
  done
done
