#!/usr/bin/env python3
"""
compute_kn_vs_weighted_norm.py
---------------------------------
Integrates your Fourier solver and learned model pipeline to compute the
mode-resolved Knudsen number (Kn) from the ratio κ_BTE / κ_Fourier and
plot it against the weighted norm w·‖G‖ of the generated conductivity map.

Workflow:
1. Load high-fidelity dataset (BTE results)
2. Select representative geometries
3. Compute κ_Fourier via your Fourier solver
4. Use your trained model to generate conductivity + adaptive coeff w
5. Compute ||G||, weighted norm, and Kn from κ_BTE / κ_Fourier
6. Plot Kn vs w·‖G‖ and save JSON + PNG outputs

Usage:
    python compute_kn_vs_weighted_norm.py <exp_name> <model_name> [--n_targets N] [--figdir DIR]
"""

import os
import sys
import json
import time
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

# ---------------------------------------------------------------------
# Import from your existing repository
# ---------------------------------------------------------------------
from solvers.low_fidelity_solvers.base_conductivity_grid_converter import conductivity_original_wrapper
from ablation_study import call_model_nojit, load_model_object   # assuming these live in your ablation utils
from solvers.low_fidelity_solvers.fourier import fourier_solver                 # replace with actual module name
# matinverse dependencies assumed available in environment


# ---------------------------------------------------------------------
# --- Knudsen number equation and solver ------------------------------
# ---------------------------------------------------------------------
def f_kn(Kn):
    """Evaluate the analytical function f(Kn) = (1 + Kn*(ln Kn - 1)) / (Kn - 1)^2"""
    Kn = np.asarray(Kn)
    out = np.full_like(Kn, np.nan, dtype=float)
    ok = Kn > 0
    k = Kn[ok]
    out[ok] = (1.0 + k * (np.log(k) - 1.0)) / ((k - 1.0) ** 2)
    return out


def solve_kn_from_ratio(R, tol=1e-8, max_iter=80):
    """
    Solve f(Kn) = R for Kn > 0.
    Uses bisection on log-spaced grid and Newton fallback.
    Returns NaN if no valid root is found.
    """
    if not np.isfinite(R) or R <= 0:
        return np.nan

    def g(k):  # root function
        return (1.0 + k * (np.log(k) - 1.0)) / ((k - 1.0) ** 2) - R

    # sample across orders of magnitude
    Ks = np.concatenate([
        np.logspace(-6, -2, 200),
        np.linspace(0.01, 0.999, 300),
        np.linspace(1.001, 10, 600),
        np.logspace(1, 6, 200)
    ])
    G = g(Ks)

    # find sign changes
    brackets = []
    for i in range(len(Ks) - 1):
        a, b = Ks[i], Ks[i + 1]
        ga, gb = G[i], G[i + 1]
        if np.isnan(ga) or np.isnan(gb):
            continue
        if ga * gb < 0:
            brackets.append((a, b))

    if not brackets:
        return np.nan

    # prefer brackets with mean > 1
    brackets.sort(key=lambda br: 0 if 0.5 * (br[0] + br[1]) > 1 else 1)
    a, b = brackets[0]
    fa, fb = g(a), g(b)

    for _ in range(max_iter):
        c = 0.5 * (a + b)
        fc = g(c)
        if not np.isfinite(fc):
            break
        if abs(fc) < tol:
            return float(c)
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
    return float(0.5 * (a + b))


# ---------------------------------------------------------------------
# --- Main computation and plotting routine ---------------------------
# ---------------------------------------------------------------------
def main(exp_name, model_name, n_targets=500, figdir=None):
    figdir = figdir or f"figures/kn_vs_weightednorm_{exp_name}_{model_name}"
    os.makedirs(figdir, exist_ok=True)

    # --- 1. Load high-fidelity dataset (BTE results)
    data = np.load("data/highfidelity/high_fidelity_2_20000.npz", allow_pickle=True)
    pores_all = np.asarray(data["pores"])
    kappas_all = np.asarray(data["kappas"]).flatten()
    if pores_all.ndim == 2 and pores_all.shape[1] == 25:
        pores_all = pores_all.reshape((-1, 5, 5))

    # --- 2. Select representative geometries
    target_kappas = np.linspace(kappas_all.min(), kappas_all.max(), num=n_targets)
    indices = np.unique([int(np.argmin(np.abs(kappas_all - t))) for t in target_kappas])
    pores_sel = pores_all[indices]
    kappas_sel = kappas_all[indices]
    print(f"[main] Selected {len(indices)} geometries")

    '''sort_idx = np.argsort(kappas_all)
    step = len(sort_idx) // n_targets
    indices = sort_idx[::step][:n_targets]  # pick evenly spaced geometries
    pores_sel = pores_all[indices]
    kappas_sel = kappas_all[indices]
    print(f"[main] Selected {len(indices)} geometries (uniformly spaced in κ)")'''

    # --- 3. Convert to conductivity grids and run Fourier solver
    print("[main] Upscaling pores to conductivity grids...")
    # Match Fourier solver input preprocessing (handle object dtype properly)
    if isinstance(pores_sel, np.ndarray) and pores_sel.dtype == object:
        pores_sel = np.stack([np.asarray(p, dtype=float) for p in pores_sel])

    grids = conductivity_original_wrapper(pores_sel, 100)

    print("[main] Running Fourier solver...")
    _, kappa_fourier = fourier_solver(jnp.asarray(grids))
    kappa_fourier = np.asarray(kappa_fourier).reshape(-1)

    # --- 4. Load trained model and generate conductivities + w
    print(f"[main] Loading model {model_name} from {exp_name} ...")
    model_obj, _ = load_model_object(exp_name, model_name)
    batch_size = 32

    cond_gen_list, w_list = [], []
    for start in range(0, len(pores_sel), batch_size):
        end = min(len(pores_sel), start + batch_size)
        batch = pores_sel[start:end]
        _, _, cond_gen, w = call_model_nojit(model_obj, batch)
        if cond_gen is not None:
            cond_gen_list.extend(list(cond_gen))
        else:
            cond_gen_list.extend([None] * (end - start))
        if w is not None:
            w_list.extend(list(np.asarray(w).flatten()))
        else:
            w_list.extend([np.nan] * (end - start))

    # --- 5. Compute Kn and weighted norms
    kn_vals, norms, ws, weighted_norms, valid_idx = [], [], [], [], []
    for i in range(len(pores_sel)):
        if cond_gen_list[i] is None or not np.isfinite(kappa_fourier[i]):
            continue
        R = kappas_sel[i] / float(kappa_fourier[i])
        Kn = solve_kn_from_ratio(R)
        if not np.isfinite(Kn):
            continue
        G = np.asarray(cond_gen_list[i]).ravel()
        normG = np.linalg.norm(G)
        w = float(w_list[i]) if np.isfinite(w_list[i]) else np.nan
        if np.isnan(w):
            continue
        kn_vals.append(Kn)
        norms.append(normG)
        ws.append(w)
        weighted_norms.append(w * normG)
        valid_idx.append(i)

    kn_vals, norms, ws, weighted_norms = map(np.array, (kn_vals, norms, ws, weighted_norms))
    print(f"[main] Computed Kn for {len(kn_vals)} geometries.")

    # --- 6. Plot Kn vs w·‖G‖
    plt.figure(figsize=(7, 6))

    # Normalize true kappas for color scale (as in ablation studies)
    kappas_norm = (kappas_sel[valid_idx] - kappas_sel[valid_idx].min()) / (
        kappas_sel[valid_idx].max() - kappas_sel[valid_idx].min()
    )

    sc = plt.scatter(
        weighted_norms,
        kn_vals,
        c=kappas_norm,
        cmap="plasma_r",
        s=24,
        edgecolors="none",
        alpha=0.85,
    )

    # Colorbar with normalized κ label
    cbar = plt.colorbar(sc)
    cbar.set_label(r"Normalized $\kappa_{\mathrm{BTE}}$", fontsize=18)

    # Linear fit
    if len(weighted_norms) > 2:
        coeffs = np.polyfit(weighted_norms, kn_vals, 1)
        a, b = coeffs
        x_fit = np.linspace(weighted_norms.min(), weighted_norms.max(), 200)
        y_fit = a * x_fit + b
        plt.plot(
            x_fit,
            y_fit,
            "k--",
            lw=2,
            #label=fr"$\mathrm{{Kn}} = {a:.3e}\,(w\Vert G\Vert) + {b:.3e}$",
        )
        plt.legend(fontsize=11, loc="best", frameon=False)
        print(f"[fit] Linear fit: Kn ≈ {a:.6e} * (w||G||) + {b:.6e}")

    # Aesthetic tweaks
    plt.xlabel(r"$w_{\phi} \cdot \Vert G \Vert$", fontsize=18)
    plt.ylabel(r"$\mathrm{Kn}$", fontsize=18)
    plt.grid(False)
    plt.tight_layout()

    out_png = os.path.join(figdir, "Kn_vs_weighted_norm.png")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"[main] Saved plot → {out_png}")

    # --- 7. Save data for further analysis
    res = {
        "indices": indices[valid_idx].tolist(),
        "kappa_BTE": kappas_sel[valid_idx].tolist(),
        "kappa_Fourier": kappa_fourier[valid_idx].tolist(),
        "Kn": kn_vals.tolist(),
        "normG": norms.tolist(),
        "w": ws.tolist(),
        "w_times_normG": weighted_norms.tolist(),
    }
    json_path = os.path.join(figdir, "Kn_vs_weighted_norm_data.json")
    with open(json_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[main] Saved results → {json_path}")


# ---------------------------------------------------------------------
# --- CLI -------------------------------------------------------------
# ---------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python compute_kn_vs_weighted_norm.py <exp_name> <model_name> [--n_targets N] [--figdir DIR]")
        sys.exit(1)
    exp_name, model_name = sys.argv[1], sys.argv[2]
    n_targets, figdir = 500, None
    for arg in sys.argv[3:]:
        if arg.startswith("--n_targets="):
            n_targets = int(arg.split("=", 1)[1])
        elif arg.startswith("--figdir="):
            figdir = arg.split("=", 1)[1]
    main(exp_name, model_name, n_targets=n_targets, figdir=figdir)
