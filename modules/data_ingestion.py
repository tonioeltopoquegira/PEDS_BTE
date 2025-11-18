import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def data_ingestion(rank, exp_name, train_size, test_size, key=42, seed=42, splits=None):
    """
    splits: None or list/tuple of two ints in {0,1,2}.
      - If None: behave like original (random split).
      - If [a,b]: train uses group a, validation uses group b (test is random).
    """
    full_data = jnp.load("data/highfidelity/high_fidelity_2_20000.npz", allow_pickle=True)
    key = key.unwrap() if hasattr(key, "unwrap") else key  # Handle JAX key from nnx

    # convenience RNG keys
    k1, k2, k3 = jrandom.split(key, 3)

    # load arrays
    pores = jnp.asarray(full_data['pores'], dtype=jnp.float32)
    kappas = jnp.asarray(full_data['kappas'], dtype=jnp.float32)
    N = kappas.shape[0]

    # helper to deterministically sample `num` indices from a 1D index-array `pool`
    def sample_from_pool(pool, num, rng):
        pool = jnp.asarray(pool)
        n = pool.shape[0]
        if n == 0:
            raise ValueError("Requested sample from empty pool.")
        if n >= num:
            perm = jrandom.permutation(rng, n)
            sel = pool[perm[:num]]
        else:
            # tile the pool enough times then permute and take first `num` (sampling with repetition)
            reps = (num + n - 1) // n
            tiled = jnp.tile(pool, reps)
            perm = jrandom.permutation(rng, tiled.shape[0])
            sel = tiled[perm[:num]]
            # warn user (use print — replace with logger if preferred)
            print(f"Warning: pool size ({n}) < requested ({num}). Sampling with repetition.")
        return sel

    if splits is None:
        # replicate original behavior: produce train_size, test_size, valid_size=test_size
        total_size = train_size + test_size
        # choose train+test+valid (train_size + test_size + test_size)
        indices = jrandom.permutation(k1, N)[:(total_size + test_size)]

        train_indices = indices[:train_size]
        test_indices = indices[train_size:total_size]          # size = test_size
        valid_indices = indices[total_size:total_size+test_size]  # size = test_size

    else:
        # Validate splits input
        if not (isinstance(splits, (list, tuple)) and len(splits) == 2):
            raise ValueError("`splits` must be None or a list/tuple of two ints (e.g. [1,2]).")
        a, b = splits
        if a not in (0, 1, 2) or b not in (0, 1, 2):
            raise ValueError("split values must be in {0,1,2}.")

        # define masks for the three groups
        group0_mask = kappas < 20.0
        group1_mask = (kappas >= 20.0) & (kappas <= 45.0)
        group2_mask = kappas > 45.0

        # arrays of indices per group
        idxs_all = jnp.arange(N)
        group_idxs = {
            0: idxs_all[group0_mask],
            1: idxs_all[group1_mask],
            2: idxs_all[group2_mask]
        }

        # 1) Sample test indices randomly from the whole dataset first
        test_indices = sample_from_pool(idxs_all, test_size, k1)

        # create remaining mask removing test indices
        mask = jnp.ones(N, dtype=bool)
        mask = mask.at[test_indices].set(False)
        remaining_idxs = idxs_all[mask]

        # For train: sample from group a but only from remaining indices belonging to that group
        group_a_pool = jnp.intersect1d(group_idxs[a], remaining_idxs, assume_unique=True)
        # For reproducibility, get a subkey for train and another for val
        k_train, k_val = jrandom.split(k2, 2)

        train_indices = sample_from_pool(group_a_pool, train_size, k_train)

        # remove train_indices from remaining
        mask2 = mask.at[train_indices].set(False)
        remaining_after_train = idxs_all[mask2]

        # For validation: sample from group b from remaining_after_train
        group_b_pool = jnp.intersect1d(group_idxs[b], remaining_after_train, assume_unique=True)
        valid_indices = sample_from_pool(group_b_pool, test_size, k_val)

        # Note: this scheme ensures train, valid, test are disjoint.

    # Create datasets
    dataset_train = [pores[train_indices], kappas[train_indices]]
    dataset_test = [pores[test_indices], kappas[test_indices]]
    dataset_valid = [pores[valid_indices], kappas[valid_indices]]

    if rank == 0:
        # For plotting convert to numpy
        train_data = np.array(kappas[train_indices])
        valid_data = np.array(kappas[valid_indices])
        test_data = np.array(kappas[test_indices])

        plt.figure(figsize=(8, 6))
        sns.kdeplot(train_data, label="Train Data", fill=True)
        sns.kdeplot(valid_data, label="Valid Data", fill=True)
        sns.kdeplot(test_data, label="Test Data", fill=True)
        plt.title("Distribution of kappas", fontsize=14)
        plt.xlabel("Kappa", fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.legend()
        plt.savefig(f"experiments/{exp_name}/figures/kappa_traintest_{seed}.png")
        plt.close()

    return dataset_train, dataset_test, dataset_valid
