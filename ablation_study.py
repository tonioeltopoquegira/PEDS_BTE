#!/usr/bin/env python3
"""
Ablation study: load a trained model and analyze generated conductivities.
Usage:
    python ablation_study.py <exp_name> <model_name> [--n_targets 500]
"""
import os
import sys
import argparse
import json
import time
from datetime import datetime

import numpy as np
import jax
import jax.numpy as jnp
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Project imports (must be on PYTHONPATH)
from models.model_utils import select_model
from modules.params_utils import initialize_or_restore_params
import config_model
from models.ensembles import ensemble

# -------------
# Utility helpers
# -------------
def _get_model_config_by_name(model_name: str):
    """Return the model_config dictionary/object from config_model by name."""
    if hasattr(config_model, model_name):
        return getattr(config_model, model_name)
    if hasattr(config_model, "models") and isinstance(getattr(config_model, "models"), dict):
        md = getattr(config_model, "models")
        if model_name in md:
            return md[model_name]
    for attr in dir(config_model):
        if attr.lower() == model_name.lower():
            return getattr(config_model, attr)
    raise KeyError(f"Model config '{model_name}' not found in config_model.py")

def ensure_batch_and_dtype_numpy(x, resolution=5, dtype=np.float32):
    """Ensure numpy array of shape (batch, res, res) and appropriate dtype"""
    a = np.asarray(x)
    if a.ndim == 1 and a.size == resolution * resolution:
        a = a.reshape((1, resolution, resolution))
    elif a.ndim == 2 and a.shape == (resolution, resolution): 
        a = a[None, ...]
    elif a.ndim == 3 and a.shape[1:] == (resolution, resolution):
        pass
    else:
        raise ValueError(f"Unsupported input shape {a.shape} for resolution {resolution}")
    if np.issubdtype(a.dtype, np.bool_) or np.issubdtype(a.dtype, np.integer):
        a = a.astype(dtype)
    return a.astype(dtype)

def load_model_object(exp_name, model_name, seed=0, rank=0, verbose=True):
    """
    Construct model object (select_model) and restore parameters from experiments/<exp_name>/weights.
    Returns the model object (ready for inference) and some info dict.
    """
    model_config = _get_model_config_by_name(model_name)
    if verbose:
        print(f"[load_model_object] Using model_config from config_model.{model_name}")

    model = select_model(
        seed=seed,
        model_type=model_config["model"],
        resolution=model_config["resolution"],
        adapt_weights=model_config.get("adapt_weights", False),
        learn_residual=model_config.get("learn_residual", False),
        hidden_sizes=model_config.get("hidden_sizes", None),
        activation=model_config.get("activation", None),
        solver=model_config.get("solver", None),
        initialization=model_config.get("initialization", None),
        n_models=model_config.get("n_models", 1),
        uq_method=model_config.get("uq_method", None),
        n_modes=model_config.get("n_modes", None),
        hidden_sizes_uq=model_config.get("hidden_sizes_uq", None),
    )

    base_dir = os.path.join("experiments", exp_name, "weights")
    model, checkpointer = initialize_or_restore_params(
        model, False, model_config["model_name"], base_dir=base_dir, rank=rank, seed=seed
    )

    info = {"restored": True, "base_dir": base_dir}
    print(f"[load_model_object] Restored model from {base_dir} (checkpointer type: {type(checkpointer)})")
    return model, info

def call_model_nojit(model_obj, pores_np):
    """
    Call model object WITHOUT jit. pores_np must be numpy array shape (batch,res,res).
    Returns (kappa_pred_np, kappa_var_np_or_None, cond_generated_np_or_None, coeff_w_np_or_None)
    """
    # ensure numpy batch shape
    resolution = pores_np.shape[1]
    x_np = ensure_batch_and_dtype_numpy(pores_np, resolution=resolution, dtype=np.float32)
    x_j = jnp.asarray(x_np)

    # If ensemble, use the first submodel for generation (consistent with training usage)
    used_model = model_obj.models[0] if isinstance(model_obj, ensemble) else model_obj

    # Call WITHOUT jit (to avoid tracer -> numpy conversion errors)
    try:
        out = used_model(x_j, training=False)
    except Exception as e:
        # Provide informative error
        raise RuntimeError("Model invocation failed (non-jit). Error: " + str(e)) from e

    # move to host
    out_host = jax.device_get(out)

    # Interpret outputs
    # Expected patterns:
    #  - (kappa, kappa_var, conductivity_generated_raw, w)  -> favored
    #  - (kappa, kappa_var, conductivity_generated_raw)  -> older behavior
    #  - kappa (single) -> fallback
    kappa_pred = None
    kappa_var = None
    cond_gen = None
    coeff_w = None

    if isinstance(out_host, (list, tuple)):
        # kappa
        kappa_pred = np.asarray(out_host[0]).reshape(-1)
        # possible var
        if len(out_host) > 1 and out_host[1] is not None:
            try:
                kappa_var = np.asarray(out_host[1]).reshape(-1)
            except Exception:
                kappa_var = None
        # conductivity
        if len(out_host) > 2 and out_host[2] is not None:
            try:
                cond_gen = np.asarray(out_host[2])
            except Exception:
                cond_gen = None
        # coefficient w (optional, shape (b,1) or (b,))
        if len(out_host) > 3 and out_host[3] is not None:
            try:
                w_arr = np.asarray(out_host[3])
                # reshape to (b,) if needed
                if w_arr.ndim == 2 and w_arr.shape[1] == 1:
                    w_arr = w_arr.reshape(-1)
                coeff_w = w_arr.reshape(-1)
            except Exception:
                coeff_w = None
    else:
        # single output (kappa)
        kappa_pred = np.asarray(out_host).reshape(-1)

    return kappa_pred, kappa_var, cond_gen, coeff_w

# --------------------------
# Main ablation study routine
# --------------------------
def main(argv):
    p = argparse.ArgumentParser()
    p.add_argument("exp_name", help="experiment folder under experiments/")
    p.add_argument("model_name", help="model config name in config_model.py")
    p.add_argument("--n_targets", type=int, default=500, help="number of target kappa samples to pick")
    p.add_argument("--figdir", type=str, default=None)
    opts = p.parse_args(argv[1:])

    exp_name = opts.exp_name
    model_name = opts.model_name
    n_targets = opts.n_targets
    figdir = opts.figdir or f"figures/ablation_{exp_name}_{model_name}"
    os.makedirs(figdir, exist_ok=True)

    print("[ablation] Loading data...")
    data = np.load("data/highfidelity/high_fidelity_2_20000.npz", allow_pickle=True)
    pores_all = np.asarray(data["pores"])
    kappas_all = np.asarray(data["kappas"]).flatten()

    # If pores stored as (N,25), reshape to (N,5,5)
    if pores_all.ndim == 2 and pores_all.shape[1] == 25:
        pores_all = pores_all.reshape((-1, 5, 5))

    N = len(kappas_all)
    print(f"[ablation] Dataset size: {N}")

    # Select representative indices (closest to evenly spaced target kappas)
    target_kappas = np.linspace(kappas_all.min(), kappas_all.max(), num=n_targets)
    selected = set()
    indices = []
    for t in target_kappas:
        idx = int(np.argmin(np.abs(kappas_all - t)))
        if idx not in selected:
            selected.add(idx)
            indices.append(idx)
    indices = np.array(indices, dtype=int)
    print(f"[ablation] selecting representative indices for {n_targets} targets ...")
    print(f"[ablation] selected {len(indices)} unique indices")

    selected_pores = pores_all[indices]
    selected_kappas = kappas_all[indices]

    # Load model object
    print(f"[ablation] attempting to load model '{model_name}' from experiment '{exp_name}' ...")
    try:
        model_obj, info = load_model_object(exp_name, model_name, seed=0, rank=0, verbose=True)
    except Exception as e:
        print("[ablation] Warning: failed to load model via load_model_object():", e)
        model_obj = None

    # If model not available, try to load a surrogate predictor (fallback)
    if model_obj is None:
        print("[ablation] No model object available. Aborting generation step.")
        # create placeholder outputs and exit gracefully
        return

    # Batch predict (call model directly no jit)
    batch_size = 128
    nsel = selected_pores.shape[0]
    kappas_pred = np.full((nsel,), np.nan, dtype=np.float32)
    kappas_var = np.full((nsel,), np.nan, dtype=np.float32)
    cond_generated_list = [None] * nsel
    coeffs_list = [None] * nsel  # store adaptive coefficient w per sample (or None)
    fail_count = 0
    t_start = time.time()
    for start in range(0, nsel, batch_size):
        end = min(nsel, start + batch_size)
        batch = selected_pores[start:end]  # numpy (b,5,5)
        try:
            t = time.time()
            kp, kv, cond, w = call_model_nojit(model_obj, batch)
            print(f"Solved batch {start}-{end} in {time.time() - t:.2f}s")
            # kp shape (b,), kv maybe (b,) or None, cond maybe (b,res,res), w maybe (b,) or None
            # store predictions
            kappas_pred[start:end] = kp
            if kv is not None:
                kappas_var[start:end] = kv
            if cond is not None:
                # ensure cond shape matches
                for i in range(cond.shape[0]):
                    cond_generated_list[start + i] = np.asarray(cond[i])
            # store w per-sample; if w is None fill with None to indicate missing
            if w is not None:
                # if w length equals batch length
                for i in range(w.shape[0]):
                    coeffs_list[start + i] = float(w[i])
            else:
                # leave as None for now
                pass
        except Exception as e:
            # keep NaNs and warn
            fail_count += (end - start)
            print(f"[ablation] Error while calling model on batch: {e}")
            # continue

    elapsed = time.time() - t_start
    print(f"[ablation] Predictions finished in {elapsed:.2f}s (n={nsel}), failed {fail_count} samples")

    # Basic quantitative metrics on selected set (if kappa preds available)
    valid_mask = ~np.isnan(kappas_pred)
    quant = {}
    if valid_mask.sum() > 0:
        preds = kappas_pred[valid_mask]
        truths = selected_kappas[valid_mask]
        quant['mse'] = float(np.mean((preds - truths) ** 2))
        quant['rmse'] = float(np.sqrt(quant['mse']))
        quant['mae'] = float(np.mean(np.abs(preds - truths)))
        print("[ablation] quantitative metrics:", quant)
    else:
        print("[ablation] No valid predictions to compute quantitative metrics.")

    # Prepare generated conductivity array (drop None)
    conds = []
    cond_truth_idx = []
    for i, c in enumerate(cond_generated_list):
        if c is not None:
            conds.append(np.asarray(c).reshape(-1))  # flatten
            cond_truth_idx.append(i)
    if len(conds) == 0:
        print("[ablation] No generated conductivity grids available; skipping qualitative plots and PCA/TSNE.")
        return

    conds = np.vstack(conds)  # shape (M, res*res)
    cond_idxs = np.array(cond_truth_idx, dtype=int)
    cond_kappas = selected_kappas[cond_idxs]
    cond_preds = kappas_pred[cond_idxs]

    # Build coefficient arrays aligned with selected indices
    coeffs_all = np.array([np.nan if v is None else float(v) for v in coeffs_list], dtype=np.float32)  # shape (nsel,)
    coeffs_for_generated = coeffs_all[cond_idxs]  # shape (M,)
    # If all NaNs, we will handle later

    # Standardize prior to PCA / t-SNE of generated conductivities
    scaler = StandardScaler()
    conds_scaled = scaler.fit_transform(conds)

    # PCA on generated conductivities (existing behavior)
    pca = PCA(n_components=2, random_state=0)
    pcs = pca.fit_transform(conds_scaled)

    fig_pca = plt.figure(figsize=(6, 5))
    # color mapping: darker = smaller true kappa
    kmin, kmax = cond_kappas.min(), cond_kappas.max()
    norm_vals = (cond_kappas - kmin) / (kmax - kmin + 1e-12)
    color_vals = norm_vals  # smaller kappa -> larger color_val
    sc = plt.scatter(pcs[:, 0], pcs[:, 1], c=color_vals, cmap="magma_r", s=20, edgecolors="none")
    cbar = plt.colorbar(sc)
    cbar.set_label("Normalized True Kappa", fontsize=14)
    #plt.title("PCA of generated conductivity (2 components)")
    plt.xlabel("PC1", fontsize=16)
    plt.ylabel("PC2", fontsize=16)
    plt.tight_layout()
    pca_path = os.path.join(figdir, "pca_generated_conductivity.png")
    fig_pca.savefig(pca_path, dpi=200)
    plt.close(fig_pca)
    print(f"[ablation] PCA plot saved to {pca_path}")

    # t-SNE on generated conductivities (existing behavior)
    tsne = TSNE(n_components=2, perplexity=30, random_state=0, init="pca")
    tsne_emb = tsne.fit_transform(conds_scaled)

    fig_tsne = plt.figure(figsize=(6, 5))
    sc2 = plt.scatter(tsne_emb[:, 0], tsne_emb[:, 1], c=color_vals, cmap="magma", s=20, edgecolors="none")
    cbar2 = plt.colorbar(sc2)
    cbar2.set_label("Inverse-normalized true kappa (darker = smaller kappa)")
    plt.title("t-SNE of generated conductivity")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    tsne_path = os.path.join(figdir, "tsne_generated_conductivity.png")
    fig_tsne.savefig(tsne_path, dpi=200)
    plt.close(fig_tsne)
    print(f"[ablation] t-SNE plot saved to {tsne_path}")

    # Scatter: true kappa vs coefficient w (only for samples where w exists)
    # replace NaN coefficients with np.nan for robust masking
    has_w_mask = ~np.isnan(coeffs_for_generated)
    scatter_path = os.path.join(figdir, "kappa_vs_coefficient_w.png")
    if np.any(has_w_mask):
        plt.figure(figsize=(6,5))
        plt.scatter(coeffs_for_generated[has_w_mask], cond_kappas[has_w_mask],
                    c=cond_preds[has_w_mask], cmap="viridis", s=30, edgecolors="none")
        plt.colorbar(label="Predicted kappa (surrogate)")
        plt.xlabel("Coefficient w")
        plt.ylabel("True kappa")
        plt.title("True kappa vs adaptive coefficient w (color=predicted kappa)")
        plt.tight_layout()
        plt.savefig(scatter_path, dpi=200)
        plt.close()
        print(f"[ablation] Scatter (kappa vs w) saved to {scatter_path}")
    else:
        print("[ablation] No coefficient w values available; skipping kappa vs w scatter. "
              "If adapt_weights was disabled during training, w may not be returned by the model.")

    # ---------------------------------------------------
    # PCA / t-SNE Ablation on pore vectors WITH and WITHOUT w
    # ---------------------------------------------------
    # Prepare pore vectors for generated set
    pores_for_generated = selected_pores[cond_idxs]  # shape (M, res, res)
    # Flatten to (M,25)
    pores_flat = pores_for_generated.reshape((pores_for_generated.shape[0], -1))

    # If coefficient vector available, create augmented 26D matrix
    # Fill missing coeffs with column mean (or zero if mean is nan)
    coeffs_col = coeffs_for_generated.reshape((-1, 1))  # shape (M,1)
    if np.all(np.isnan(coeffs_col)):
        # no coefficients at all -> fill with zeros
        coeffs_filled = np.zeros_like(coeffs_col)
        coeffs_note = "all_missing_filled_with_zero"
    else:
        mean_val = float(np.nanmean(coeffs_col))
        coeffs_filled = np.nan_to_num(coeffs_col, nan=mean_val)
        coeffs_note = f"nan_filled_with_mean_{mean_val:.6g}"

    X25 = pores_flat.copy()
    X26 = np.hstack([pores_flat, coeffs_filled])

    # Standardize (separately for fairness)
    scaler25 = StandardScaler()
    X25_s = scaler25.fit_transform(X25)

    scaler26 = StandardScaler()
    X26_s = scaler26.fit_transform(X26)

    # PCA 25
    pca25 = PCA(n_components=2, random_state=0)
    pcs25 = pca25.fit_transform(X25_s)
    pca25_path = os.path.join(figdir, "pca_pores_25d.png")
    plt.figure(figsize=(6,5))
    sc25 = plt.scatter(pcs25[:,0], pcs25[:,1], c=cond_kappas, cmap="viridis", s=20, edgecolors="none")
    plt.colorbar(sc25, label="True kappa")
    plt.title("PCA on pores (25D)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(pca25_path, dpi=200)
    plt.close()
    print(f"[ablation] PCA (25D pores) saved to {pca25_path}")

    # PCA 26
    pca26 = PCA(n_components=2, random_state=0)
    pcs26 = pca26.fit_transform(X26_s)
    pca26_path = os.path.join(figdir, "pca_pores_26d_with_w.png")
    plt.figure(figsize=(6,5))
    sc26 = plt.scatter(pcs26[:,0], pcs26[:,1], c=cond_kappas, cmap="viridis", s=20, edgecolors="none")
    plt.colorbar(sc26, label="True kappa")
    plt.title("PCA on pores + coefficient w (26D)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(pca26_path, dpi=200)
    plt.close()
    print(f"[ablation] PCA (26D pores + w) saved to {pca26_path}")

    # t-SNE 25
    tsne25 = TSNE(n_components=2, perplexity=30, random_state=0, init="pca")
    tsne25_emb = tsne25.fit_transform(X25_s)
    tsne25_path = os.path.join(figdir, "tsne_pores_25d.png")
    plt.figure(figsize=(6,5))
    sc_tsne25 = plt.scatter(tsne25_emb[:,0], tsne25_emb[:,1], c=cond_kappas, cmap="viridis", s=20, edgecolors="none")
    plt.colorbar(sc_tsne25, label="True kappa")
    plt.title("t-SNE on pores (25D)")
    plt.xlabel("t-SNE1")
    plt.ylabel("t-SNE2")
    plt.tight_layout()
    plt.savefig(tsne25_path, dpi=200)
    plt.close()
    print(f"[ablation] t-SNE (25D pores) saved to {tsne25_path}")

    # t-SNE 26
    tsne26 = TSNE(n_components=2, perplexity=30, random_state=0, init="pca")
    tsne26_emb = tsne26.fit_transform(X26_s)
    tsne26_path = os.path.join(figdir, "tsne_pores_26d_with_w.png")
    plt.figure(figsize=(6,5))
    sc_tsne26 = plt.scatter(tsne26_emb[:,0], tsne26_emb[:,1], c=cond_kappas, cmap="viridis", s=20, edgecolors="none")
    plt.colorbar(sc_tsne26, label="True kappa")
    plt.title("t-SNE on pores + coefficient w (26D)")
    plt.xlabel("t-SNE1")
    plt.ylabel("t-SNE2")
    plt.tight_layout()
    plt.savefig(tsne26_path, dpi=200)
    plt.close()
    print(f"[ablation] t-SNE (26D pores + w) saved to {tsne26_path}")

        # Qualitative: plot conductivity grids at fixed target kappas
    target_vals = [11.0, 25.0, 45.0, 160.0]
    pick_idxs = []

    for tval in target_vals:
        # find index with closest true kappa
        diffs = np.abs(selected_kappas - tval)
        idx = np.argmin(diffs)
        if cond_generated_list[idx] is not None:
            pick_idxs.append(idx)

    # ensure we got 4 (fallback if some missing)
    while len(pick_idxs) < 4:
        rnd = np.random.choice(cond_idxs)
        if rnd not in pick_idxs and cond_generated_list[rnd] is not None:
            pick_idxs.append(rnd)

    # Shared color scale
    vmin = min(np.min(cond_generated_list[i]) for i in pick_idxs)
    vmax = max(np.max(cond_generated_list[i]) for i in pick_idxs)

    fig, axes = plt.subplots(1, len(pick_idxs), figsize=(4 * len(pick_idxs), 4), constrained_layout=True)
    if len(pick_idxs) == 1:
        axes = [axes]

    ims = []
    for ax, idx in zip(axes, pick_idxs):
        g = np.asarray(cond_generated_list[idx])
        if g.ndim == 1:
            res = int(np.sqrt(g.size))
            g = g.reshape((res, res))

        im = ax.imshow(g, origin="lower", vmin=vmin, vmax=vmax, cmap="viridis")
        ims.append(im)

        true_k = selected_kappas[idx]
        pred_k = kappas_pred[idx] if not np.isnan(kappas_pred[idx]) else None
        wval = coeffs_all[idx]
        wstr = "n/a" if np.isnan(wval) else f"{wval:.4f}"

        ax.set_title(f"Kappa ~{true_k:.2f}\n Mixing Coefficient {wstr}", fontsize=16)
        ax.axis("off")

    # one shared colorbar
    cbar = fig.colorbar(ims[0], ax=axes, orientation="vertical", fraction=0.025, pad=0.04)
    cbar.set_label("Conductivity", fontsize=14)

    qualitative_path = os.path.join(figdir, "example_generated_conductivities.png")
    fig.suptitle(f"Example conductivity grids at targets [11, 25, 45, 160] ({exp_name}/{model_name})")
    fig.savefig(qualitative_path, dpi=200)
    plt.close(fig)
    print(f"[ablation] Example conductivity plot saved to {qualitative_path}")



    # ---- Compute "max horizontal pore run length" for each geometry ----
    # (Assumes original pores were stored in 5x5 before upscaling.)
    max_run_per_geom = []
    for pores in selected_pores[cond_idxs]:
        grid = (pores > 0.5).astype(int)  # ensure binary
        max_run = 0
        for row in grid:
            # consecutive 1s per row
            run = 0
            for val in row:
                if val == 1:
                    run += 1
                    max_run = max(max_run, run)
                else:
                    run = 0
        max_run_per_geom.append(max_run)
    max_run_per_geom = np.array(max_run_per_geom)

    # ---- Scatter plots: max run vs PCA1 and PCA2 ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)

    axes[0].scatter(max_run_per_geom, pcs[:, 0], c=cond_kappas, cmap="viridis", alpha=0.7, s=20)
    axes[0].set_xlabel("Max horizontal pore run length")
    axes[0].set_ylabel("PCA1")
    axes[0].set_title("Max run vs PCA1")

    axes[1].scatter(max_run_per_geom, pcs[:, 1], c=cond_kappas, cmap="viridis", alpha=0.7, s=20)
    axes[1].set_xlabel("Max horizontal pore run length")
    axes[1].set_ylabel("PCA2")
    axes[1].set_title("Max run vs PCA2")

    plt.tight_layout()
    runlen_path = os.path.join(figdir, "max_run_vs_pca.png")
    fig.savefig(runlen_path, dpi=200)
    plt.close(fig)
    print(f"[ablation] Max-run vs PCA plots saved to {runlen_path}")

    
    # Extra geometry statistics (5x5 pore grids) and PCA correlations
    

    def connected_components_and_spanning(grid):
        """Return list of component sizes, and booleans for horizontal and vertical spanning."""
        H, W = grid.shape
        visited = np.zeros_like(grid, dtype=bool)
        comps = []
        horiz_span = False
        vert_span = False
        from collections import deque
        for i in range(H):
            for j in range(W):
                if grid[i, j] == 1 and not visited[i, j]:
                    q = deque()
                    q.append((i, j))
                    visited[i, j] = True
                    size = 0
                    touches_left = touches_right = touches_top = touches_bottom = False
                    while q:
                        x, y = q.popleft()
                        size += 1
                        if y == 0: touches_left = True
                        if y == W - 1: touches_right = True
                        if x == 0: touches_top = True
                        if x == H - 1: touches_bottom = True
                        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < H and 0 <= ny < W and not visited[nx, ny] and grid[nx, ny] == 1:
                                visited[nx, ny] = True
                                q.append((nx, ny))
                    comps.append(size)
                    if touches_left and touches_right:
                        horiz_span = True
                    if touches_top and touches_bottom:
                        vert_span = True
        return comps, horiz_span, vert_span

    def euler_number_from_binary(grid):
        """Compute Euler number (components - holes) via 2x2 block patterns."""
        H, W = grid.shape
        euler = 0
        for i in range(H - 1):
            for j in range(W - 1):
                block = grid[i:i+2, j:j+2].astype(int)
                s = block.sum()
                if s == 1:
                    euler += 1
                elif s == 3:
                    euler -= 1
        return int(euler)

    # Compute statistics
    res = 5
    mean_neighbors_arr = []
    total_edges_arr = []
    largest_comp_arr = []
    n_components_arr = []
    euler_arr = []

    for pores in selected_pores[cond_idxs]:
        grid = (pores > 0.5).astype(int)

        # neighbor counts
        neigh_counts = []
        edges = 0
        for i in range(res):
            for j in range(res):
                if grid[i, j] == 1:
                    c = 0
                    for di, dj in [(1,0),(-1,0),(0,1),(0,-1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < res and 0 <= nj < res and grid[ni, nj] == 1:
                            c += 1
                    neigh_counts.append(c)
                    edges += c
        mean_neighbors_arr.append(np.mean(neigh_counts) if neigh_counts else 0.0)
        total_edges_arr.append(edges // 2)

        comps, _, _ = connected_components_and_spanning(grid)
        n_components_arr.append(len(comps))
        largest_comp_arr.append(max(comps) if comps else 0)
        euler_arr.append(euler_number_from_binary(grid))

    mean_neighbors_arr = np.array(mean_neighbors_arr)
    total_edges_arr = np.array(total_edges_arr)
    largest_comp_arr = np.array(largest_comp_arr)
    n_components_arr = np.array(n_components_arr)
    euler_arr = np.array(euler_arr)

    # Build DataFrame
    import pandas as pd
    df_stats = pd.DataFrame({
        "mean_neighbors": mean_neighbors_arr,
        "total_edges": total_edges_arr,
        "largest_comp": largest_comp_arr,
        "n_components": n_components_arr,
        "euler": euler_arr,
        "PC1": pcs[:, 0],
        "PC2": pcs[:, 1],
        "kappa": cond_kappas
    })

    print("[ablation] Correlation matrix (stats vs PCA):")
    print(df_stats.corr()[["PC1","PC2"]].round(3))

    # Plot scatter comparisons
    plot_stats = ["mean_neighbors", "total_edges", "largest_comp", "n_components", "euler"]
    fig, axes = plt.subplots(len(plot_stats), 2, figsize=(10, 4*len(plot_stats)))

    for r, stat in enumerate(plot_stats):
        x = df_stats[stat].values
        axes[r, 0].scatter(x, df_stats["PC1"], c=df_stats["kappa"], cmap="viridis", s=20, alpha=0.7)
        axes[r, 0].set_xlabel(stat)
        axes[r, 0].set_ylabel("PC1")
        axes[r, 0].set_title(f"{stat} vs PC1")

        axes[r, 1].scatter(x, df_stats["PC2"], c=df_stats["kappa"], cmap="viridis", s=20, alpha=0.7)
        axes[r, 1].set_xlabel(stat)
        axes[r, 1].set_ylabel("PC2")
        axes[r, 1].set_title(f"{stat} vs PC2")

    plt.tight_layout()
    stats_path = os.path.join(figdir, "geom_stats_vs_pca.png")
    fig.savefig(stats_path, dpi=200)
    plt.close(fig)
    print(f"[ablation] Geometry statistics vs PCA plots saved to {stats_path}")

    # Save raw table
    csv_path = os.path.join(figdir, "geom_stats_table.csv")
    df_stats.to_csv(csv_path, index=False)
    print(f"[ablation] Geometry stats table saved to {csv_path}")

        # ---- New analysis: relationship between conductivity norm, w, and true thermal conductivity ----
    norms = []
    ws = []
    kappas_true_eff = []

    for idx in cond_idxs:
        g = cond_generated_list[idx]
        if g is None:
            continue
        g = np.asarray(g)
        norms.append(np.linalg.norm(g))
        ws.append(coeffs_all[idx])
        kappas_true_eff.append(cond_kappas[np.where(cond_idxs == idx)[0][0]])  # use true κ

    norms = np.array(norms)
    ws = np.array(ws)
    kappas_true_eff = np.array(kappas_true_eff)

    if len(norms) > 0:
        fig, ax = plt.subplots(figsize=(6, 5))

        # --- normalize kappas for consistent colormap ---
        kmin, kmax = kappas_true_eff.min(), kappas_true_eff.max()
        norm_vals = (kappas_true_eff - kmin) / (kmax - kmin + 1e-12)
        color_vals = norm_vals   # darker = smaller kappa (like PCA block)

        sc = ax.scatter(ws * norms, kappas_true_eff, 
                        c=color_vals, cmap="magma_r", s=20, edgecolors="none")

        ax.set_xlabel(r"$w_{\phi} \cdot ||\mathrm{G}||$", fontsize=14)
        ax.set_ylabel("Effective Thermal Conductivity", fontsize=14)

        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Normalized True Kappa", fontsize=14)

        path = os.path.join(figdir, "weighted_norm_vs_true_kappa.png")
        fig.savefig(path, dpi=200)
        plt.close(fig)
        print(f"[ablation] Weighted norm vs true κ plot saved to {path}")

        # optional correlation matrix
        import pandas as pd
        df = pd.DataFrame({
            "norm": norms,
            "w": ws,
            "w*norm": ws * norms,
            "kappa_true": kappas_true_eff
        })
        corr = df.corr()
        print("[ablation] correlation matrix (true κ):\n", corr)






    # Save summary JSON (expanded)
    summary = {
        "exp_name": exp_name,
        "model_name": model_name,
        "n_selected": len(indices),
        "n_generated": len(cond_idxs),
        "quantitative": quant,
        "pca_generated_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "pca_pores_25d_explained_variance_ratio": pca25.explained_variance_ratio_.tolist(),
        "pca_pores_26d_explained_variance_ratio": pca26.explained_variance_ratio_.tolist(),
        "coeffs_note": coeffs_note,
        "coeffs_stats": {
            "mean": None if np.all(np.isnan(coeffs_for_generated)) else float(np.nanmean(coeffs_for_generated)),
            "min": None if np.all(np.isnan(coeffs_for_generated)) else float(np.nanmin(coeffs_for_generated)),
            "max": None if np.all(np.isnan(coeffs_for_generated)) else float(np.nanmax(coeffs_for_generated)),
            "n_present": int(np.sum(~np.isnan(coeffs_for_generated)))
        },
        "paths": {
            "pca_generated": pca_path,
            "tsne_generated": tsne_path,
            "kappa_vs_w_scatter": scatter_path if os.path.exists(scatter_path) else None,
            "pca_pores_25d": pca25_path,
            "pca_pores_26d": pca26_path,
            "tsne_pores_25d": tsne25_path,
            "tsne_pores_26d": tsne26_path,
            "qualitative": qualitative_path
        },
        "timestamp": datetime.now().isoformat(),
    }
    summary_path = os.path.join(figdir, "ablation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[ablation] Summary saved to {summary_path}")
    print("[ablation] Done.")

if __name__ == "__main__":
    main(sys.argv)
