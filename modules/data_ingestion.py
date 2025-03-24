import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def data_ingestion(rank, model_name, filename, total_size, train_size, stratified, key=42,filename_highfid=None):


    full_data = jnp.load("data/highfidelity/" + filename, allow_pickle=True)
    # Generate random indices
    key = key.unwrap() if hasattr(key, "unwrap") else key  # Extract JAX key if it's an nnx RngStream

    indices = jrandom.permutation(key, len(full_data['pores']))[:total_size]

    # Use the indices to sample from the data
    pores_all = jnp.asarray(full_data['pores'], dtype=jnp.float32)
    kappas_all = jnp.asarray(full_data['kappas'], dtype=jnp.float32)

    pores = pores_all[indices]
    kappas = kappas_all[indices]
    design_valid_indices = None

    if stratified == "all":

        indices = jrandom.permutation(key, jnp.arange(total_size))
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
        design_valid_indices = indices[total_size:total_size+40] #probably want to stratify
    
    if stratified =="small->small":
       
        small_indices = jnp.where(kappas_all < 80)[0]
        shuffled_small_indices = jrandom.permutation(key, small_indices)[:total_size]
        shuffled_small_indices = jrandom.permutation(key, shuffled_small_indices)


        train_indices = shuffled_small_indices[:train_size]
        test_indices = shuffled_small_indices[train_size:]
        design_valid_indices = shuffled_small_indices[total_size:total_size+40]

        fidelity = jnp.ones(total_size)
        dataset_train = [pores_all[train_indices], kappas_all[train_indices], fidelity[train_indices]]
        dataset_valid = [pores_all[test_indices], kappas_all[test_indices], fidelity[test_indices]]
        kappas_design_valid = kappas_all[design_valid_indices]

        if rank == 0:

            # Plot prob distribution of kappa for train and test and save it to train_results


            # Assuming pores[train_indices] and pores[test_indices] are already defined
            train_data = np.array(kappas_all[train_indices])
            test_data = np.array(kappas_all[test_indices])

            # Create the plot
            plt.figure(figsize=(8, 6))

            # Plot the KDE for the training data
            sns.kdeplot(train_data, label="Train Data", color="blue", fill=True)

            # Plot the KDE for the test data
            sns.kdeplot(test_data, label=f"Test Data", color="red", fill=True)

            # Add labels and title
            plt.title("Distribution of kappas", fontsize=14)
            plt.xlabel("Pore Values", fontsize=12)
            plt.ylabel("Density", fontsize=12)
            plt.legend()

            # Show the plot
            plt.savefig(f"figures/models/{model_name}/kappa_distribution_traintest.png")
            plt.close()

        return dataset_train, dataset_valid, kappas_design_valid


    elif stratified == "small->big":
        
        # Stratify by sorting kappas, selecting top `train_size` sorted points for training
        sorted_indices = jnp.argsort(kappas)
        train_indices = sorted_indices[:train_size]
        test_indices = sorted_indices[train_size:]

    
    elif stratified =="big->small":

        # Stratify by sorting kappas, selecting top `train_size` sorted points for training
        sorted_indices = jnp.argsort(kappas)
        train_indices = sorted_indices[train_size:]
        test_indices = sorted_indices[:train_size]

    
    fidelity = jnp.ones(total_size)
    dataset_train = [pores[train_indices], kappas[train_indices], fidelity[train_indices]]
    dataset_valid = [pores[test_indices], kappas[test_indices], fidelity[test_indices]]
    kappas_valid = kappas[design_valid_indices]

    if rank == 0:

        # Plot prob distribution of kappa for train and test and save it to train_results


        # Assuming pores[train_indices] and pores[test_indices] are already defined
        train_data = np.array(kappas[train_indices])
        test_data = np.array(kappas[test_indices])

        # Create the plot
        plt.figure(figsize=(8, 6))

        # Plot the KDE for the training data
        sns.kdeplot(train_data, label="Train Data", color="blue", fill=True)

        # Plot the KDE for the test data
        sns.kdeplot(test_data, label=f"Test Data", color="red", fill=True)

        # Add labels and title
        plt.title("Distribution of kappas", fontsize=14)
        plt.xlabel("Pore Values", fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.legend()

        # Show the plot
        plt.savefig(f"figures/models/{model_name}/kappa_distribution_traintest.png")
        plt.close()

    return dataset_train, dataset_valid, kappas_valid