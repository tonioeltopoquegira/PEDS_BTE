# surrogate_utils.py
import os
import numpy as np
import jax
import jax.numpy as jnp

# project-specific imports (match your training script)
from models.model_utils import predict as model_predict
from models.model_utils import plot_example  # optional, unused here
from models.ensembles import ensemble        # for isinstance checks
from models.model_utils import predict
from models.ensembles import ensemble

from models.model_utils import predict
from models.ensembles import ensemble

from models.model_utils import predict
from models.ensembles import ensemble

from modules.params_utils import initialize_or_restore_params
from models.model_utils import select_model

# config_model should exist in your repo as in main()
import config_model


def _ensure_batch_and_dtype(x, resolution=5, dtype=jnp.float32):
    """Ensure x has shape (batch, resolution, resolution) and float dtype."""
    x = np.asarray(x)
    if x.ndim == 1 and x.size == resolution * resolution:
        x = x.reshape((1, resolution, resolution))
    elif x.ndim == 2 and x.shape == (resolution, resolution):
        x = x[None, ...]
    elif x.ndim == 3 and x.shape[1:] == (resolution, resolution):
        pass
    else:
        raise ValueError(f"Unsupported input shape {x.shape} for resolution {resolution}")
    # booleans/ints -> floats
    if np.issubdtype(x.dtype, np.bool_) or np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float32)
    return jnp.asarray(x, dtype=dtype)


def _get_model_config_by_name(model_name: str):
    """Return the model_config dictionary/object from config_model by name."""
    # same approach used earlier in your main: config_model exports variables named like peds_fourier etc.
    if hasattr(config_model, model_name):
        return getattr(config_model, model_name)
    # try dictionary style
    if hasattr(config_model, "models") and isinstance(getattr(config_model, "models"), dict):
        models_dict = getattr(config_model, "models")
        if model_name in models_dict:
            return models_dict[model_name]
    # fallback by case-insensitive attribute match
    for attr in dir(config_model):
        if attr.lower() == model_name.lower():
            return getattr(config_model, attr)
    raise KeyError(f"Model config '{model_name}' not found in config_model.py")


def load_surrogate_predictor(
    exp_name: str,
    model_name: str,
    seed: int = 0,
    rank: int = 0,
    restore: bool = True,
    verbose: bool = True,
):
    """
    Load model configured in config_model.<model_name> from experiments/<exp_name>/weights
    and return a fast callable predict_surrogate(pores).

    pores can be:
      - a (5,5) array
      - a flat length-25 vector
      - a batch (n,5,5)

    The returned callable returns a scalar kappa (float) for single input, or a numpy array
    for a batch.
    """
    # 1) find the model config
    model_config = _get_model_config_by_name(model_name)
    if verbose:
        print(f"[surrogate_utils] Using model_config from config_model.{model_name}")

    # 2) construct the model with the same args your main uses
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

    # 3) restore params using the same helper your training code uses
    base_dir = os.path.join("experiments", exp_name, "weights")
    # main() typically calls initialize_or_restore_params(model, False, model_config["model_name"], base_dir=..., rank=rank, seed=seed)
    model, checkpointer = initialize_or_restore_params(
        model, False, model_config["model_name"], base_dir=base_dir, rank=rank, seed=seed
    )

    if verbose:
        print(f"[surrogate_utils] initialize_or_restore_params returned model type={type(model)}, checkpointer={type(checkpointer)}")

    # 4) build a small wrapper around your models.model_utils.predict function
    resolution = int(model_config["resolution"])

    @jax.jit
    def _predict_jit(x_batch):
        """
        x_batch: jnp array of shape (N, 5, 5), dtype float32
        Returns: (kappa_pred, kappa_var)
        """
        out = model_predict(model, x_batch)
        if isinstance(out, tuple) or isinstance(out, list):
            kappa_pred, kappa_var = out
        else:
            kappa_pred, kappa_var = out, None
        return kappa_pred, kappa_var

    # Warm-up JIT so first call isn't slow
    _ = _predict_jit(jnp.zeros((1, resolution, resolution), dtype=jnp.float32))

    def predict_surrogate(pores, *, return_var=False):
        """
        pores: single (5,5) geometry or batch
        return_var: return variance estimate if available
        """
        x = _ensure_batch_and_dtype(pores, resolution=resolution)

        # JIT call
        kappa_pred, kappa_var = _predict_jit(x)

        # Move back to numpy
        kp = np.asarray(jax.device_get(kappa_pred))
        if kp.ndim > 1 and kp.shape[-1] == 1:
            kp = kp.reshape(kp.shape[:-1])

        # single sample → scalar
        if kp.size == 1:
            scalar = float(kp.reshape(-1)[0])
            if not return_var:
                return scalar
            if kappa_var is None:
                return scalar, None
            kv = np.asarray(jax.device_get(kappa_var))
            return scalar, float(kv.reshape(-1)[0])
        
        # batch case
        if return_var:
            if kappa_var is None:
                return kp.reshape(-1), None
            kv = np.asarray(jax.device_get(kappa_var))
            return kp.reshape(-1), kv.reshape(-1)

        return kp.reshape(-1)

    if verbose:
        print(f"[surrogate_utils] Surrogate predictor ready for model '{model_name}' (exp: {exp_name}).")

    return predict_surrogate
