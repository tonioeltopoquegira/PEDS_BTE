#!/usr/bin/env bash
# run_splits_models.sh
# Sequentially run all split experiments x models using 4 MPI processes.
# Each job prints to stdout/stderr (no redirection to logs).

set -u



exps=(
  "split_0_1"
  "split_0_2"
  "split_1_2"
  "split_1_0"
  "split_2_0"
  "split_2_1"
)

models=(
  "peds_fourier_uq1"
  "peds_fourier"
  "mlpmod"
  "mlp_ens_uq1"
)

pyenv activate peds
echo "Starting all combinations of experiments and models..."

for exp in "${exps[@]}"; do
  for model in "${models[@]}"; do
    echo "=== Running: exp='${exp}' model='${model}' ==="
    mpiexec -n 4 python pipeline_peds.py "${exp}" "${model}"
    rc=$?
    echo "Exit code: ${rc}"
    echo "----------------------------------------"
    # continue to next combination even if one fails
  done
done

echo "All combinations finished."
