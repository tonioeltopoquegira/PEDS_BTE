This repository implements code for data-efficient inverse design of heat conductive nano-materials using Physics Enhanced Deep Surrogates and Uncertainty Quantification. 
The main experiments were implemented using MatInverse differentiable fourier solver that is not yet released. 
For this reason we have creted a separate branch 'experiments' were you can replicate the results using a Gauss Seidel solver or direct solver.
To replicate the results duplicate the 'experiments' branch and then:
1. The requirements are can be installed in a virtual environment just by simply running 'pip install -r requirements.txt' in the terminal
2. Run 'python full_pipeline.py <experiment name> <model name>'       or on multiple CPUs 'mpiexec -n 4 full_pipeline.py <experiment name> <model name>' 

A list of the experiments and model names can be found below, while the actual configurations can be found in config_experiments.py and config_model.py


experiments = {
  "1000_data": basic_1000_train, <- Basic experiment, 1000 training points, 100 test points
  "100_data": dataeff_100_train, <- 100 training points, 100 test points
  "200_data": dataeff_200_train, <- 200 training points, 100 test points
  "500_data": dataeff_500_train, <- 500 training points, 100 test points
  "earlystop": earlystop, <- Active learning for 100 size case
}

models = {
  "peds_fourier": peds_fourier,
  "peds_gauss": peds_gauss,
  "mlpmod": mlpmod,
  "peds_fourier_ens": peds_f_ens,
  "peds_gauss_ens": peds_g_ens,
  "peds_fourier_uq1": peds_f_ens_uq1,
  "peds_gauss_uq1": peds_g_ens_uq1
}

Note that you can just run code with peds_gauss or peds_g models!! 
