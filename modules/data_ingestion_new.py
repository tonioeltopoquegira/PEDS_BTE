import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def data_ingestion(rank, exp_name, train_size, test_size, key=42):
    full = jnp.load("data/highfidelity/high_fidelity_2_20000.npz", allow_pickle=True)
    #data/highfidelity/high_fidelity_2_20000.npz
    # unwrap nnx key if necessary
    rng = key.unwrap() if hasattr(key, "unwrap") else key

    pores = jnp.asarray(full['pores'], dtype=jnp.float32)
    kappas = jnp.asarray(full['kappas'], dtype=jnp.float32)
    N = len(pores)

    # 1) Shuffle all indices
    perm = jrandom.permutation(rng, N)
    
    # 2) Allocate first `train_size` to training
    train_idx = perm[:train_size]

    # 3) Pool the remainder for stratified test & validation
    pool_idx = np.array(perm[train_size:])  # to NumPy for boolean masking
    pool_kappa = np.array(kappas)[pool_idx]

    # Define your strata boundaries (in same units as `kappas`)
    strata_bounds = [15, 40]  # percentages
    # Create masks and lists of indices per stratum
    strata = {
        '<15':   pool_idx[pool_kappa <  strata_bounds[0]],
        '15-40': pool_idx[(pool_kappa >= strata_bounds[0]) & (pool_kappa <  strata_bounds[1])],
        '>40':   pool_idx[pool_kappa >= strata_bounds[1]],
    }

    # Helper to sample from a numpy array with JAX RNG
    def sample_npy(arr, k, rng):
        if len(arr) == 0:
            return np.array([], dtype=int), rng
        # draw a permutation of indices
        subperm = jrandom.permutation(rng, len(arr))
        rng, _ = jrandom.split(rng)
        sel = np.array(arr)[np.array(subperm[:k])]
        return sel, rng

    # 4) Guarantee at least one from each stratum in both test & valid  
    test_idx = []
    valid_idx = []
    for bucket in strata.values():
        if len(bucket) >= 2:
            # take one for test, one for valid
            sel = jrandom.choice(rng, bucket, shape=(2,), replace=False)
            rng, _ = jrandom.split(rng)
            test_idx.append(int(sel[0]))
            valid_idx.append(int(sel[1]))
        elif len(bucket) == 1:
            # only one sample: put it in test, leave valid for later fill
            test_idx.append(int(bucket[0]))
    
    test_idx = np.array(test_idx, dtype=int)
    valid_idx = np.array(valid_idx, dtype=int)

    # Remove already‐chosen from pool
    used = set(test_idx.tolist() + valid_idx.tolist())
    remaining = np.array([i for i in pool_idx if i not in used], dtype=int)

    # 5) Fill up to `test_size` and `test_size` for valid
    need_test  = test_size  - len(test_idx)
    need_valid = test_size  - len(valid_idx)

    add_test,  rng = sample_npy(remaining, need_test,  rng)
    remaining = remaining[np.isin(remaining, add_test, invert=True)]
    add_valid, rng = sample_npy(remaining, need_valid, rng)

    test_idx  = np.concatenate([test_idx,  add_test])
    valid_idx = np.concatenate([valid_idx, add_valid])

    # Convert to JAX arrays
    train_idx = jnp.array(train_idx, dtype=jnp.int32)
    test_idx  = jnp.array(test_idx,  dtype=jnp.int32)
    valid_idx = jnp.array(valid_idx, dtype=jnp.int32)

    dataset_train = [pores[train_idx],  kappas[train_idx]]
    dataset_test  = [pores[test_idx],   kappas[test_idx]]
    dataset_valid = [pores[valid_idx], kappas[valid_idx]]

    if rank == 0:
        # Plot distributions
        train_k = np.array(kappas)[train_idx]
        test_k  = np.array(kappas)[test_idx]
        plt.figure(figsize=(8, 6))
        sns.kdeplot(train_k, label="Train", fill=True)
        sns.kdeplot(test_k,  label="Test",  fill=True)
        plt.title("Distribution of kappas")
        plt.xlabel("kappa value")
        plt.ylabel("Density")
        plt.legend()
        plt.savefig(f"experiments/{exp_name}/figures/kappa_traintest_{rng}.png")
        plt.close()

    return dataset_train, dataset_test, dataset_valid
