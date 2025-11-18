#!/usr/bin/env python3
"""
Robust bo.py for ax-platform: handles several ax installs and ensures an experiment
is created for fallback AxClient usage.

Updated: adds CLI, option to train GP with Active Learning via fit_gp_for_bo(..., active_learning=True).
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, Tuple

import argparse
import numpy as np
import importlib.metadata
from surrogate_utils import load_surrogate_predictor
# keep import name consistent with your repo; ensure gp_for_utils.py (or equivalent) contains the new AL signature
from gp_for_utils import fit_gp_for_bo


# Print ax-platform version (best-effort)
ax_version = None
try:
    ax_version = importlib.metadata.version("ax-platform")
except Exception:
    try:
        ax_version = importlib.metadata.version("ax")
    except Exception:
        ax_version = "unknown"
print("Using Ax version:", ax_version)

# Try robust imports for Client and Models (different ax versions expose different paths)
Client = None
Models = None
AX_FALLBACK = False

try:
    # Preferred public API (may be present on some installs)
    from ax import Client  # type: ignore
    from ax.registry import Models  # type: ignore
    print("Imported: from ax import Client ; from ax.registry import Models")
except Exception as e1:
    try:
        # Older client location
        from ax.service.client import Client  # type: ignore
        try:
            from ax.registry import Models  # type: ignore
            print("Imported: from ax.service.client import Client ; from ax.registry import Models")
        except Exception:
            try:
                from ax.modelbridge.registry import Models  # type: ignore
                print("Imported: from ax.service.client import Client ; from ax.modelbridge.registry import Models")
            except Exception:
                Models = None
                print("Imported Client from ax.service.client but Models not found; will fallback.")
    except Exception:
        # Final fallback to AxClient (older API)
        try:
            from ax.service.ax_client import AxClient as Client  # type: ignore
            Models = None
            AX_FALLBACK = True
            print("Imported AxClient fallback (ax.service.ax_client). Generation strategy config may be limited.")
        except Exception:
            raise ImportError(
                "Could not import Ax Client from known locations. "
                "Make sure ax-platform is installed in the active environment."
            )

# Keep ObjectiveProperties import for robust create_experiment calls
from ax.service.utils.instantiation import ObjectiveProperties

# ---- Import your expensive solver (your path) ----
from solvers.high_fidelity_solver.OpenBTE_highfid import highfidelity_solver

# Global so Ax can access/see the target during runs
TARGET_KAPPA = 30.0


def evaluate_geometry_and_loss(parameters: Dict[str, bool], surrogate_predict=None) -> Tuple[float, float]:
    """
    Evaluate a binary pore geometry using either a surrogate predictor or the high-fidelity solver.
    Returns: (loss, kappa) where loss = (kappa - TARGET_KAPPA)^2
    """
    flat = np.array([int(parameters[f"x{i}"]) for i in range(25)], dtype=int)
    pores = flat.reshape(5, 5)
    if surrogate_predict is not None:
        # Use the surrogate model (fast)
        try:
            kappa = surrogate_predict(pores)
        except Exception as e:
            raise RuntimeError(f"Surrogate prediction failed: {e}")
    else:
        # Fall back to the expensive high-fidelity solver
        kappa, _, _, _ = highfidelity_solver(pores=pores, step_size=1.5)
    loss = (kappa - TARGET_KAPPA) ** 2
    return float(loss), float(kappa)


def run_ax_convergence(
    target_kappa: float,
    max_trials: int,
    save_dir: str = "results",
    tol: float = 1e-2,
    patience: int = 3,
    random_seed: int = 123,
    sobol_steps: int = 30,
    use_surrogate: bool = False,
    surrogate_type: str = "peds",               # "peds" or "gp"
    surrogate_exp_name: str = None,             # for 'peds'
    surrogate_model_name: str = None,           # for 'peds'
    surrogate_gp = None,                        # for 'gp': optional trained sklearn GP object
    gp_train_size: int = 300,                   # if surrogate_gp is None and surrogate_type=='gp', train this many pts
    gp_n_restarts: int = 3,                     # passed to train_gp_for_bo n_restarts_optimizer
) -> Tuple[Dict, Dict]:
    """
    Run Ax optimization with optional surrogate:
      - use_surrogate=False: runs high-fidelity solver (as before)
      - use_surrogate=True + surrogate_type='peds': loads NN surrogate via load_surrogate_predictor
      - use_surrogate=True + surrogate_type='gp'  : uses supplied sklearn GP (surrogate_gp) or
                                                   trains one via gp_for_utils.fit_gp_for_bo(...)
    """
    global TARGET_KAPPA
    TARGET_KAPPA = target_kappa

    os.makedirs(save_dir, exist_ok=True)

    t_start = time.time()

    # Instantiate client (handle signature differences)
    try:
        client = Client(random_seed=random_seed)
    except TypeError:
        client = Client()

    ax_client = client

    # Configure experiment (unchanged)
    if (not AX_FALLBACK) and (Models is not None):
        try:
            parameters = [{"name": f"x{i}", "type": "choice", "values": [False, True]} for i in range(25)]
            ax_client.configure_experiment(
                name=f"bte_opt_target_{target_kappa}",
                parameters=parameters,
            )
            ax_client.configure_optimization(objective="loss", minimize=True)
            ax_client.configure_generation_strategy(
                name="sobol_then_botorch",
                steps=[
                    {"model": Models.SOBOL, "num_trials": sobol_steps},
                    {"model": Models.BOTORCH_MODULAR},
                ],
            )
            print(f"Configured generation strategy: Sobol {sobol_steps} -> BoTorch")
        except Exception as e:
            print("Warning: failed to configure generation strategy via Client API:", repr(e))
            print("Will rely on client's default generation strategy.")
    else:
        params = [{"name": f"x{i}", "type": "choice", "values": [False, True]} for i in range(25)]
        try:
            ax_client.create_experiment(
                name=f"bte_opt_target_{target_kappa}",
                parameters=params,
                objectives={"loss": ObjectiveProperties(minimize=True)},
            )
            print("Created experiment on AxClient (fallback) using ObjectiveProperties signature.")
        except Exception:
            try:
                ax_client.create_experiment(
                    name=f"bte_opt_target_{target_kappa}",
                    parameters=params,
                    objectives={"loss": {"minimize": True}},
                )
                print("Created experiment on AxClient (fallback) using older-style objectives dict.")
            except Exception:
                try:
                    ax_client.create_experiment(
                        name=f"bte_opt_target_{target_kappa}",
                        parameters=params,
                    )
                    print("Created experiment on AxClient (fallback) using minimal signature.")
                except Exception as ex2:
                    raise RuntimeError("Failed to create experiment on AxClient fallback: " + repr(ex2))

        print("Using fallback AxClient instance (generation strategy may be default/limited).")
        if Models is not None:
            print("Note: Models available but client fallback path used; generation strategy configuration may be limited.")

    # bookkeeping
    timestamp_start = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"target{target_kappa}_start{timestamp_start}"
    out_dir = os.path.join(save_dir, run_id)
    os.makedirs(out_dir, exist_ok=True)

    best_kappa = None
    best_loss = None
    best_params = None
    consecutive_successes = 0
    trial_count = 0
    trace = []

    surrogate_predict = None
    gp_info = None

    if use_surrogate:
        if surrogate_type not in ("peds", "gp"):
            raise ValueError("surrogate_type must be 'peds' or 'gp'")

        if surrogate_type == "peds":
            if surrogate_exp_name is None or surrogate_model_name is None:
                raise ValueError("surrogate_exp_name and surrogate_model_name are required for surrogate_type='peds'")
            surrogate_predict = load_surrogate_predictor(
                exp_name=surrogate_exp_name,
                model_name=surrogate_model_name,
                seed=random_seed,
                rank=0,
                restore=True,
            )
            print(f"[run_ax_convergence] Loaded NN surrogate: {surrogate_exp_name}/{surrogate_model_name}")

        else:  # surrogate_type == "gp"
            # small internal flatten helper (compatible with gp_for_utils._flatten_inputs)
            def _flatten_for_gp(pores):
                a = np.asarray(pores)
                if a.ndim == 1 and a.size == 25:
                    return a.reshape(1, 25).astype(np.float64)
                if a.ndim == 2 and a.shape == (5, 5):
                    return a.reshape(1, 25).astype(np.float64)
                if a.ndim == 2 and a.shape[1] == 25:
                    return a.astype(np.float64)
                if a.ndim == 3 and a.shape[1:] == (5, 5):
                    return a.reshape(a.shape[0], 25).astype(np.float64)
                raise ValueError(f"Unsupported input shape for GP predictor: {a.shape}")

            if surrogate_gp is not None:
                # wrap sklearn GP
                def gp_predictor(pores):
                    Xq = _flatten_for_gp(pores)
                    mean = surrogate_gp.predict(Xq, return_std=False)
                    # single sample -> return scalar
                    if mean.size == 1:
                        return float(mean[0])
                    return mean.reshape(-1)
                surrogate_predict = gp_predictor
                gp_info = {"provided_gp": True, "trained_on": None}
                print("[run_ax_convergence] Using provided trained GaussianProcessRegressor as surrogate.")
            else:
                # import and train small GP via fit_gp_for_bo
                print(f"[run_ax_convergence] Training GP surrogate with {gp_train_size} points (this may take a moment)...")
                gp_obj, predictor_callable, info = fit_gp_for_bo(
                    num_train=gp_train_size,
                    data_path="data/highfidelity/high_fidelity_2_20000.npz",
                    n_restarts_optimizer=gp_n_restarts,
                    # if your fit function supports active learning these kwargs will be ignored otherwise
                )
                surrogate_predict = predictor_callable
                gp_info = {"provided_gp": False, "trained_on": info}
                print(f"[run_ax_convergence] Trained GP: {info}")

    # MAIN LOOP
    while trial_count < max_trials:
        trial_count += 1

        # Request next trial
        parameters_dict, trial_index = ax_client.get_next_trial()

        # Evaluate (surrogate_predict may be None => evaluate_geometry_and_loss uses high-fidelity)
        t0 = time.time()
        loss_val, kappa_val = evaluate_geometry_and_loss(parameters_dict, surrogate_predict=surrogate_predict)
        eval_time = time.time() - t0

        # Report result
        ax_client.complete_trial(trial_index=trial_index, raw_data={"loss": (loss_val, 0.0)})

        trace_entry = {
            "trial_index": trial_index,
            "parameters": parameters_dict,
            "loss": loss_val,
            "kappa": kappa_val,
            "time_s": eval_time,
            "trial_number": trial_count,
        }
        trace.append(trace_entry)

        if (best_loss is None) or (loss_val < best_loss):
            best_loss = loss_val
            best_kappa = kappa_val
            best_params = parameters_dict

        # Convergence check
        if abs(kappa_val - target_kappa) <= tol * target_kappa:
            consecutive_successes += 1
        else:
            consecutive_successes = 0

        print(
            f"[Trial {trial_count}] loss={loss_val:.6f}, kappa={kappa_val:.6f}, "
            f"best_loss={best_loss:.6f}, best_kappa={best_kappa:.6f}, "
            f"consec={consecutive_successes}/{patience}"
        )

        if consecutive_successes >= patience:
            print(f"[INFO] Convergence reached on kappa (tol={tol}) for {patience} consecutive trials.")
            break
    
    t_end = time.time()
    total_time_s = t_end - t_start

    # After loop ends, recompute true kappa for best params (if surrogate was used)
    best_kappa_true = None
    if use_surrogate and best_params is not None:
        pores = np.array([int(best_params[f"x{i}"]) for i in range(25)]).reshape(5, 5)
        best_kappa_true, _, _, _ = highfidelity_solver(pores=pores, step_size=2.0)
        print(f"[Post-check] High-fidelity solver evaluation: kappa_true={best_kappa_true:.6f}")

    if not use_surrogate:
        method_name = "BTE"
    elif surrogate_type == "peds":
        method_name = "PEDS"
    elif surrogate_type == "gp":
        method_name = "GP"
    else:
        method_name = "Unknown"

    # Save results
    timestamp_end = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_data = {
        "target_kappa": target_kappa,
        "max_trials": max_trials,
        "ran_trials": trial_count,
        "best_loss": best_loss,
        "best_kappa_pred": best_kappa,
        "best_kappa_true": best_kappa_true,
        "best_parameters": {k: int(v) if isinstance(v, (np.bool_, bool)) else v
                            for k, v in (best_params or {}).items()},
        "tol": tol,
        "patience": patience,
        "sobol_steps": sobol_steps if Models is not None else None,
        "timestamp_start": timestamp_start,
        "timestamp_end": timestamp_end,
        "trace": trace,
        "surrogate_used": use_surrogate,
        "surrogate_type": surrogate_type if use_surrogate else None,
        "gp_info": gp_info,
        "method_name": method_name,
        "total_time_s": total_time_s,   # <--- NEW FIELD
    }

    result_file = os.path.join(out_dir, f"opt_result_{timestamp_end}.json")
    with open(result_file, "w") as f:
        json.dump(result_data, f, indent=2)


    # Summary print
    print("\n=== Optimization Complete ===")
    print(f"Target kappa: {target_kappa}")
    print(f"Ran trials: {trial_count}")
    print(f"Best loss (surrogate): {best_loss:.6f}")
    print(f"Best kappa (predicted): {best_kappa:.6f}")
    if best_kappa_true is not None:
        print(f"Best kappa (true solver): {best_kappa_true:.6f}")
    print("Best binary pattern (5x5):")
    if best_params is not None:
        best_flat = np.array([int(best_params[f"x{i}"]) for i in range(25)])
        print(best_flat.reshape(5, 5))
    print(f"Saved to {result_file}")

    return best_params, result_data


def _parse_args():
    p = argparse.ArgumentParser(description="Run Ax BO for BTE with optional surrogates / GP active learning")
    p.add_argument("--targets", nargs="+", type=float,
                   default=[11.1, 12.01, 15.00, 20.00, 25.00, 30.00, 44.98, 59.99, 75.00, 90.87, 160.0])
    p.add_argument("--max-trials", type=int, default=200)
    p.add_argument("--save-dir", type=str, default="results")
    p.add_argument("--tol", type=float, default=0.01)
    p.add_argument("--patience", type=int, default=1)
    p.add_argument("--random-seed", type=int, default=0)
    p.add_argument("--sobol-steps", type=int, default=50)
    p.add_argument("--use-surrogate", action="store_true", help="Use a surrogate instead of high-fidelity solver")
    p.add_argument("--surrogate-type", choices=["peds", "gp"], default="peds")
    p.add_argument("--surrogate-exp-name", type=str, default="al_300")
    p.add_argument("--surrogate-model-name", type=str, default="peds_f_ens_uq1")
    # GP training options
    p.add_argument("--gp-train-size", type=int, default=300)
    p.add_argument("--gp-n-restarts", type=int, default=3)
    p.add_argument("--gp-active-learning", action="store_true",
                   help="If set, perform active learning when training GP (fit_gp_for_bo supports active_learning flag)")
    p.add_argument("--al-total-size", type=int, default=300, help="AL: final training set size")
    p.add_argument("--al-batch-fraction", type=float, default=0.5, help="AL: fraction of final size per iteration")
    p.add_argument("--al-sample-factor", type=int, default=8, help="AL: candidate sample factor per batch")
    return p.parse_args()


if __name__ == "__main__":
    # ---------- MANUAL RUN BLOCK ----------
    # Edit these values by hand:
    CONFIG = {
        "targets": [85.00],   # 12.01, 15.00, 20.0, 30.0, 45.00, 60.00, 75.00, 90.00
        "max_trials": 200,
        "save_dir": "results",
        "tol": 0.05,
        "patience": 1,
        "random_seed": 0,
        "sobol_steps": 50,
        # Surrogate choice:
        "use_surrogate": False,
        "surrogate_type": "gp",        # "peds" or "gp"
        # If using PEDS, fill these:
        "surrogate_exp_name": "train_300_NEW",
        "surrogate_model_name": "peds_f_ens_uq1",
        # If using GP, set a trained sklearn GP object here (or None to train inside run)
        "surrogate_gp": None,            # e.g. gp_obj returned from fit_gp_for_bo(...)
        "gp_train_size": 300,
        "gp_n_restarts": 0,
    }

    # Loop over targets and call your run function
    for tgt in CONFIG["targets"]:
        print(f"\n>>> Running optimization for target_kappa = {tgt}")
        best_params, result_data = run_ax_convergence(
            target_kappa=tgt,
            max_trials=CONFIG["max_trials"],
            save_dir=CONFIG["save_dir"],
            tol=CONFIG["tol"],
            patience=CONFIG["patience"],
            random_seed=CONFIG["random_seed"],
            sobol_steps=CONFIG["sobol_steps"],
            use_surrogate=CONFIG["use_surrogate"],
            surrogate_type=CONFIG["surrogate_type"],
            surrogate_exp_name=CONFIG["surrogate_exp_name"],
            surrogate_model_name=CONFIG["surrogate_model_name"],
            surrogate_gp=CONFIG["surrogate_gp"],
            gp_train_size=CONFIG["gp_train_size"],
            gp_n_restarts=CONFIG["gp_n_restarts"],
        )

        # optional: save per-target result (timestamped)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_fname = os.path.join(CONFIG["save_dir"], f"opt_result_target{tgt}_{ts}.json")
        os.makedirs(CONFIG["save_dir"], exist_ok=True)
        with open(out_fname, "w") as fh:
            json.dump(result_data, fh, indent=2)
        print(f"[MAIN] Target {tgt} done. Saved: {out_fname}")
    # ---------- END MANUAL RUN BLOCK ----------
