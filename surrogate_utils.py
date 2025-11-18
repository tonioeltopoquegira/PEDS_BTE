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

    def predict_surrogate(pores, *, return_var=False):
        """
        pores: (5,5) or length-25 or batch (n,5,5)
        return_var: if True and the model returns variance, return (kappa, var)
        """
        x = _ensure_batch_and_dtype(pores, resolution=resolution)

        # model_predict is your project-level predict(...) function used in training/validation.
        # Its typical uses in your code:
        #   kappa_pred, kappa_var = predict(model, val_pores)
        #   kappa_pred, kappa_var = predict(model, val_pores, training=True, ... ) during train
        # For inference we call predict(model, x) without training flag.
        try:
            out = model_predict(model, x)  # call your project predict wrapper
        except TypeError:
            # Some predict wrappers require explicit training flag
            try:
                out = model_predict(model, x, training=False)
            except Exception as e:
                raise RuntimeError(f"Could not call model_predict(...). Original error: {e}")

        # model_predict may return (kappa_pred, kappa_var) or just kappa_pred
        if isinstance(out, tuple) or isinstance(out, list):
            kappa_pred = out[0]
            kappa_var = out[1] if len(out) > 1 else None
        else:
            kappa_pred = out
            kappa_var = None

        kappa_np = np.asarray(jax.device_get(kappa_pred))
        # reduce singleton dimensions (e.g., (1,) -> scalar)
        if kappa_np.ndim > 1 and kappa_np.shape[-1] == 1:
            kappa_np = kappa_np.reshape(kappa_np.shape[:-1])
        if kappa_np.size == 1:
            scalar = float(kappa_np.reshape(-1)[0])
            if return_var:
                if kappa_var is None:
                    return scalar, None
                var_np = np.asarray(jax.device_get(kappa_var))
                return scalar, float(var_np.reshape(-1)[0]) if var_np.size == 1 else var_np.reshape(-1)
            return scalar

        # batch case: return numpy array, optionally with variance
        if return_var:
            var_np = np.asarray(jax.device_get(kappa_var)) if kappa_var is not None else None
            return kappa_np.reshape(-1), (var_np.reshape(-1) if var_np is not None else None)

        return kappa_np.reshape(-1)

    if verbose:
        print(f"[surrogate_utils] Surrogate predictor ready for model '{model_name}' (exp: {exp_name}).")

    return predict_surrogate
