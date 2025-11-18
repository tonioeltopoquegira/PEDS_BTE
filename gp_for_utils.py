import numpy as np
from typing import Tuple, Dict, Callable
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.metrics import mean_squared_error


DEFAULT_DATA_PATH = "data/highfidelity/high_fidelity_2_20000.npz"


def _flatten_inputs(p):
    """Ensure pores are flattened into (N,25) float64."""
    a = np.asarray(p)
    if a.ndim == 1:
        if a.size != 25:
            raise ValueError("1D input must have length 25")
        return a.reshape(1,25).astype(np.float64)
    if a.ndim == 2:
        if a.shape == (5,5):
            return a.reshape(1,25).astype(np.float64)
        if a.shape[1] == 25:
            return a.astype(np.float64)
        raise ValueError(f"Unsupported 2D shape {a.shape}")
    if a.ndim == 3 and a.shape[1:] == (5,5):
        return a.reshape(a.shape[0],25).astype(np.float64)
    raise ValueError(f"Unsupported input shape {a.shape}")


def fit_gp_for_bo(
    num_train: int = 300,
    data_path: str = DEFAULT_DATA_PATH,
    test_size: int = 1000,
    train_slice_start: int = 1000,
    seed: int = 43,
    n_restarts_optimizer: int = 3,
    # Active learning options
    active_learning: bool = False,
    al_total_size: int = 2000,
    al_batch_fraction: float = 0.5,
    al_sample_factor: int = 8,
) -> Tuple[object, Callable, Dict]:
    """
    Fit a GaussianProcessRegressor on high-fidelity dataset.

    If active_learning is False (default) this behaves like the original:
      - draws `num_train` examples from the shuffled dataset (starting at
        train_slice_start) and fits a GP.

    If active_learning is True, perform an uncertainty-based active learning
    loop to gather `al_total_size` training points from the data pool. The
    AL loop mimics the procedure in your snippet: sample a candidate subset
    from the pool, predict std, and select the most uncertain examples.

    Returns: (gp, predictor_callable, info_dict)
    """

    # Load dataset
    data = np.load(data_path, allow_pickle=True)
    X_all = np.asarray(data["pores"]).reshape(-1,25).astype(np.float64)
    y_all = np.asarray(data["kappas"]).flatten().astype(np.float64)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X_all))

    # Create test split
    test_idx = perm[:test_size]
    pool_idx = perm[test_size:]

    X_test = X_all[test_idx]
    y_test = y_all[test_idx]

    # If not using AL, follow original sampling strategy
    if not active_learning:
        available = pool_idx
        if num_train > len(available):
            num_train = len(available)
        # pick a slice starting at train_slice_start (clamp)
        start = int(train_slice_start)
        if start < 0:
            start = 0
        if start >= len(available):
            start = 0
        end = start + num_train
        # wrap-around if necessary
        if end <= len(available):
            train_idx = available[start:end]
        else:
            # if requested slice overruns, just take the first num_train available
            train_idx = available[:num_train]

        X_train = X_all[train_idx]
        y_train = y_all[train_idx]

    else:
        # Active learning flow: initialize empty training set and a pool
        X_pool = X_all[pool_idx].copy()
        y_pool = y_all[pool_idx].copy()

        X_train = np.empty((0, X_all.shape[1]), dtype=np.float64)
        y_train = np.empty((0,), dtype=np.float64)

        total_size = int(al_total_size)
        batch_fraction = float(al_batch_fraction)
        sample_factor = int(al_sample_factor)

        # Safety checks
        if total_size <= 0:
            raise ValueError("al_total_size must be > 0")
        if batch_fraction <= 0 or batch_fraction > 1:
            raise ValueError("al_batch_fraction must be in (0, 1]")
        if sample_factor < 1:
            raise ValueError("al_sample_factor must be >= 1")

        iteration = 0
        # Continue until we have enough training points (or pool exhausted)
        while len(X_train) < total_size and len(X_pool) > 0:
            iteration += 1
            remaining = total_size - len(X_train)
            batch_size = int(max(1, min(int(total_size * batch_fraction), remaining)))
            sample_size = int(min(len(X_pool), batch_size * sample_factor))

            # Sample candidate indices from the pool
            sample_idx = rng.choice(len(X_pool), size=sample_size, replace=False)
            X_candidates = X_pool[sample_idx]
            y_candidates = y_pool[sample_idx]

            if len(X_train) > 0:
                # Fit GP on current training set
                kernel = Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=1e-6)
                gp_iter = GaussianProcessRegressor(
                    kernel=kernel,
                    normalize_y=True,
                    random_state=seed,
                    n_restarts_optimizer=max(0, int(n_restarts_optimizer)),
                )
                gp_iter.fit(X_train, y_train)

                # Predict uncertainty on candidates and pick most uncertain
                _, std = gp_iter.predict(X_candidates, return_std=True)
                uncertain_local_idx = np.argsort(-std)[:batch_size]
            else:
                # first iteration: choose first batch_size candidates (deterministic)
                uncertain_local_idx = np.arange(min(batch_size, len(X_candidates)))

            # Map local candidate indices back to pool indices
            selected_pool_idx = sample_idx[uncertain_local_idx]

            # Add selected to training set
            X_sel = X_pool[selected_pool_idx]
            y_sel = y_pool[selected_pool_idx]
            X_train = np.vstack([X_train, X_sel])
            y_train = np.concatenate([y_train, y_sel])

            # Remove selected from pool (delete by indices)
            # Need to sort descending so indices remain valid when deleting one-by-one via np.delete
            to_delete = np.unique(selected_pool_idx)
            X_pool = np.delete(X_pool, to_delete, axis=0)
            y_pool = np.delete(y_pool, to_delete, axis=0)

            print(f"[AL it={iteration}] Training set size: {len(X_train)} / {total_size} (pool remaining: {len(X_pool)})")

        # If we exited because pool exhausted but didn't reach total_size, we just use what we have
        if len(X_train) == 0:
            raise RuntimeError("Active learning produced no training points (pool empty).")

    # -------------------------
    # Fit final GP on X_train
    # -------------------------
    kernel = Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=1e-6)
    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        random_state=seed,
        n_restarts_optimizer=max(0, int(n_restarts_optimizer)),
    )
    gp.fit(X_train, y_train)

    # Eval
    y_pred, y_std = gp.predict(X_test, return_std=True)
    mse = float(mean_squared_error(y_test, y_pred))

    info = {
        "num_train": len(X_train),
        "num_test": len(X_test),
        "mse_test": mse,
        "kernel_learned": str(gp.kernel_),
        "active_learning": bool(active_learning),
    }

    if active_learning:
        info.update({
            "al_total_size": al_total_size,
            "al_batch_fraction": al_batch_fraction,
            "al_sample_factor": al_sample_factor,
        })

    def predictor(pores, return_std: bool = False):
        Xq = _flatten_inputs(pores)
        if return_std:
            mean, std = gp.predict(Xq, return_std=True)
            if mean.size == 1:
                return float(mean[0]), float(std[0])
            return mean, std
        else:
            mean = gp.predict(Xq, return_std=False)
            if mean.size == 1:
                return float(mean[0])
            return mean

    return gp, predictor, info
