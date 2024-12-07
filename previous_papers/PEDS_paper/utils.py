import jax

def choose_nonlinearity(name):
    """
    Returns the appropriate non-linearity function based on the given name.

    Args:
        name (str): The name of the non-linearity function.

    Returns:
        function: The corresponding non-linearity function.
    """
    if name == 'tanh':
        nl = jax.nn.tanh
    elif name == 'relu':
        nl = jax.nn.relu
    elif name == 'sigmoid':
        nl = jax.nn.sigmoid
    elif name == 'softplus':
        nl = jax.nn.softplus
    elif name == 'selu':
        nl = lambda x: jax.nn.selu(x)
    elif name == 'elu':
        nl = jax.nn.elu
    elif name == 'swish':
        nl = lambda x: x * jax.nn.sigmoid(x)
    else:
        raise ValueError("Nonlinearity not recognized")
    return nl