This repository implements code for data-efficient inverse design of heat conductive nano-materials using Physics Enhanced Deep Surrogates and Uncertainty Quantification. 
The main experiments were implemented using MatInverse differentiable fourier solver that is not yet released. 
For this reason we have creted a separate branch 'experiments' were you can replicate the results using a Gauss Seidel solver or direct solver.
To replicate the results duplicate the 'experiments' branch and then:
1. The requirements are can be installed in a virtual environment just by simply running 'pip install -r requirements.txt' in the terminal
2. Run 'python full_pipeline.py <experiment name> <model name>'       or on multiple CPUs 'mpiexec -n 4 full_pipeline.py <experiment name> <model name>' 

A list of the experiments and model names can be found below, while the actual configurations can be found in config_experiments.py and config_model.py

