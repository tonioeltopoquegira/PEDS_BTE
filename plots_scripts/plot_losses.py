import numpy as np
import os
import matplotlib.pyplot as plt

# Function to plot losses with additional metrics like variance and percentual error
def plot_losses(models_data, experiment_name):
    """
    Plots training and validation losses over epochs for multiple models,
    along with the average variance (standard deviation) and percentual error.

    Parameters:
    - models_data: List of dictionaries, each containing:
        - 'train_losses': List of training loss values.
        - 'val_losses': List of validation loss values.
        - 'epochs': Number of epochs (integer).
        - 'model_name': Name of the model (string).
        - 'valid_variance': List of validation variance values.
        - 'valid_perc_losses': List of validation percentual errors.
    - experiment_name: Name of the experiment (string).
    """
    plt.figure(figsize=(12, 8))
    
    for model_data in models_data:
        train_losses = model_data['train_losses']
        val_losses = np.sqrt(model_data['val_losses']) # Adjust to per observation (assuming 100 observations)
        epochs = model_data['epochs']
        model_name = model_data['model_name']
        valid_variance = model_data['valid_variance']
        valid_perc_losses = model_data['valid_perc_losses']
        
        # Plot Training and Validation Losses
        #plt.plot(range(1, epochs + 1), train_losses, label=f'{model_name} - Training Loss', marker='o')
        plt.plot(range(1, epochs + 1), val_losses, label=f'{model_name} - Validation Loss')
        
        # Plot Validation Loss with Standard Deviation as "confidence interval"
        std_dev = np.sqrt(valid_variance)  # Standard deviation from variance
        plt.fill_between(range(1, epochs + 1), val_losses - std_dev, val_losses + std_dev, alpha=0.2, color='blue')
        
        # Plot Percentual Loss
        plt.plot(range(1, epochs + 1), valid_perc_losses, label=f'{model_name} - Percentual Error', linestyle='--')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{experiment_name} - Loss Curves')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

# Main function to load data and plot
if __name__ == "__main__":
    # Specify experiment name and model names
    experiment_name = "opt_coding"
    model_names = ["peds_ensemble"]

    models_data = []  # List to store the data for each model

    # Load data for each model and store in models_data list
    for model_name in model_names:
        curves = np.load(f"experiments/{experiment_name}/curves/training_curves_{model_name}.npz", allow_pickle=True)
        
        model_data = {
            'train_losses': curves['epoch_losses'],
            'val_losses': curves['valid_losses'],
            'valid_variance': curves['valid_variance'],  # Assuming the variance is available in the dataset
            'valid_perc_losses': curves['valid_perc_losses'],  # Assuming the percentual errors are available
            'epochs': len(curves['epoch_losses']),
            'model_name': model_name
        }
        
        models_data.append(model_data)

    # Plot the losses for all models
    plot_losses(models_data, experiment_name)
