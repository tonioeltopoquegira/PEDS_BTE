import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def data_ingestion(rank, exp_name, train_size, test_size, key=42, seed=42):
    full_data = jnp.load("data/highfidelity/high_fidelity_2_20000.npz", allow_pickle=True)
    key = key.unwrap() if hasattr(key, "unwrap") else key  # Handle JAX key from nnx

    total_size = train_size + test_size 

    indices = jrandom.permutation(key, len(full_data['pores']))[:total_size+test_size]  # Shuffle indices

    # Load data
    pores = jnp.asarray(full_data['pores'], dtype=jnp.float32)
    kappas = jnp.asarray(full_data['kappas'], dtype=jnp.float32)
    
    # Assign train and test based on stratification
    
    #indices = jrandom.permutation(key, jnp.arange(total_size+test_size))
    train_indices = indices[:train_size]
    test_indices = indices[train_size:total_size]
    valid_indices = indices[total_size:total_size+test_size]

    # Create datasets
    dataset_train = [pores[train_indices], kappas[train_indices]]
    dataset_test = [pores[test_indices], kappas[test_indices]]
    dataset_valid = [pores[valid_indices], kappas[valid_indices]]

    '''data = np.load("data/highfidelity/high_fidelity_NEW_2_5000.npz", allow_pickle=True)
    pores = data["pores"]  # expected shape (N, 25)
    kappas = data["kappas"]  # expected shape (N,)

    pores = pores / 20.0  # simple normalization


    # Ensure numeric types (float32)
    pores = np.array(pores, dtype=np.float32)
    kappas = np.array(kappas, dtype=np.float32)

    # Filter out kappas outside [0, 160]
    kappas_mask = (kappas >= 0) & (kappas <= 160)
    pores = pores[kappas_mask]
    kappas = kappas[kappas_mask]

    # Convert to JAX arrays
    pores = jnp.asarray(pores)
    kappas = jnp.asarray(kappas)

    total_size = train_size + test_size
    valid_size = test_size  # assuming test_size is also your validation size, adjust if different

    # Create a PRNGKey if input key is int, else unwrap if needed
    if isinstance(key, int):
        key = jrandom.PRNGKey(key)
    elif hasattr(key, "unwrap"):
        key = key.unwrap()

    # Shuffle indices for total_size + valid_size (train + test + val)
    total_indices = pores.shape[0]
    if total_indices < total_size + valid_size:
        raise ValueError(f"Not enough data after filtering: got {total_indices}, need {total_size + valid_size}")

    indices = jrandom.permutation(key, total_indices)

    # Select indices for train, test, valid
    train_indices = indices[:train_size]
    test_indices = indices[train_size:train_size + test_size]
    valid_indices = indices[train_size + test_size:train_size + test_size + valid_size]

    # Create datasets
    dataset_train = [pores[train_indices], kappas[train_indices]]
    dataset_test = [pores[test_indices], kappas[test_indices]]
    dataset_valid = [pores[valid_indices], kappas[valid_indices]]'''

    if rank == 0:

        # Plot prob distribution of kappa for train and test and save it to train_results


        # Assuming pores[train_indices] and pores[test_indices] are already defined
        train_data = np.array(kappas[train_indices])
        valid_data = np.array(kappas[valid_indices])
        test_data = np.array(kappas[test_indices])


        # Create the plot
        plt.figure(figsize=(8, 6))

        # Plot the KDE for the training data
        sns.kdeplot(train_data, label="Train Data", color="blue", fill=True)

        sns.kdeplot(valid_data, label="Valid Data", color="green", fill=True)

        # Plot the KDE for the test data
        sns.kdeplot(test_data, label=f"Test Data", color="red", fill=True)

        # Add labels and title
        plt.title("Distribution of kappas", fontsize=14)
        plt.xlabel("Pore Values", fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.legend()

        # Show the plot
        plt.savefig(f"experiments/{exp_name}/figures/kappa_traintest_{seed}.png")
        plt.close()

    return dataset_train, dataset_test, dataset_valid