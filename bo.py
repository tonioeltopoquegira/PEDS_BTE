#!/usr/bin/env python3
"""
Robust bo.py for ax-platform: handles several ax installs and ensures an experiment
is created for fallback AxClient usage.
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import importlib.metadata

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


def evaluate_geometry_and_loss(parameters: Dict[str, bool]) -> Tuple[float, float]:
    """
    Evaluate a binary pore geometry using the high-fidelity solver.
    Returns: (loss, kappa) where loss = (kappa - TARGET_KAPPA)^2
    """
    flat = np.array([int(parameters[f"x{i}"]) for i in range(25)], dtype=int)
    pores = flat.reshape(5, 5)
    kappa, _, _ = highfidelity_solver(pores=pores, step_size=2.0)
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
) -> Tuple[Dict, Dict]:
    """
    Run Ax optimization with optional Sobol warm-start (if Models available).
    - If Models is available, configures Sobol->BoTorch generation strategy.
    - Otherwise uses the AxClient fallback behavior; we ensure create_experiment is called.
    - Convergence = |kappa - target_kappa| <= tol * target_kappa for `patience` consecutive trials.
    """

    global TARGET_KAPPA
    TARGET_KAPPA = target_kappa

    os.makedirs(save_dir, exist_ok=True)

    # Instantiate client (handle signature differences)
    try:
        client = Client(random_seed=random_seed)
    except TypeError:
        # Some Client constructors have different signatures
        client = Client()

    ax_client = client

    # If we have Models, configure experiment + generation strategy via Client API
    if (not AX_FALLBACK) and (Models is not None):
        try:
            # Define parameters
            parameters = [{"name": f"x{i}", "type": "choice", "values": [False, True]} for i in range(25)]
            ax_client.configure_experiment(
                name=f"bte_opt_target_{target_kappa}",
                parameters=parameters,
            )
            ax_client.configure_optimization(objective="loss", minimize=True)
            # Configure Sobol warmup -> BoTorch (if Models available)
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
        # AX_FALLBACK: ensure experiment is created on AxClient
        # AxClient.create_experiment signature varies, use robust calls
        params = [{"name": f"x{i}", "type": "choice", "values": [False, True]} for i in range(25)]
        try:
            # preferred modern signature
            ax_client.create_experiment(
                name=f"bte_opt_target_{target_kappa}",
                parameters=params,
                objectives={"loss": ObjectiveProperties(minimize=True)},
            )
            print("Created experiment on AxClient (fallback) using ObjectiveProperties signature.")
        except Exception:
            # try older signature
            try:
                ax_client.create_experiment(
                    name=f"bte_opt_target_{target_kappa}",
                    parameters=params,
                    objectives={"loss": {"minimize": True}},
                )
                print("Created experiment on AxClient (fallback) using older-style objectives dict.")
            except Exception as ex:
                # Last resort: attempt minimal create_experiment call
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

    # BOOKKEEPING
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

    # MAIN LOOP: sequential single-trial evaluations
    while trial_count < max_trials:
        trial_count += 1

        # Request next trial (works for both Client and AxClient)
        parameters_dict, trial_index = ax_client.get_next_trial()

        # Evaluate expensive black-box
        t0 = time.time()
        loss_val, kappa_val = evaluate_geometry_and_loss(parameters_dict)
        eval_time = time.time() - t0

        # Report result
        ax_client.complete_trial(trial_index=trial_index, raw_data={"loss": (loss_val, 0.0)})

        # Log
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

        # Convergence check (on reported kappa)
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

    # Save results
    timestamp_end = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_data = {
        "target_kappa": target_kappa,
        "max_trials": max_trials,
        "ran_trials": trial_count,
        "best_loss": best_loss,
        "best_kappa": best_kappa,
        "best_parameters": {k: int(v) if isinstance(v, (np.bool_, bool)) else v for k, v in (best_params or {}).items()},
        "tol": tol,
        "patience": patience,
        "sobol_steps": sobol_steps if Models is not None else None,
        "timestamp_start": timestamp_start,
        "timestamp_end": timestamp_end,
        "trace": trace,
    }
    result_file = os.path.join(out_dir, f"opt_result_{timestamp_end}.json")
    with open(result_file, "w") as f:
        json.dump(result_data, f, indent=2)

    # Summary print
    print("\n=== Optimization Complete ===")
    print(f"Target kappa: {target_kappa}")
    print(f"Ran trials: {trial_count}")
    print(f"Best loss: {best_loss:.6f}")
    print(f"Best kappa: {best_kappa:.6f}")
    print("Best binary pattern (5x5):")
    if best_params is not None:
        best_flat = np.array([int(best_params[f"x{i}"]) for i in range(25)])
        print(best_flat.reshape(5, 5))
    print(f"Saved to {result_file}")

    return best_params, result_data


if __name__ == "__main__":
    targets = [12.01, 15.00, 20.00, 30.00, 44.98, 59.99]
    trials_2 = 500

    for t in targets:
        run_ax_convergence(
            target_kappa=t,
            max_trials=trials_2,
            save_dir="results",
            tol=0.01,
            patience=1,
            random_seed=43,
            sobol_steps=50,  # only applies if Models config path works; otherwise None
        )
