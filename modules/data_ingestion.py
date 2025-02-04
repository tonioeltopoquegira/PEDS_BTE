import jax.numpy as jnp
import jax.random as jrandom

def data_ingestion(filename, train_size, stratified, multifidelity,key=42,filename_highfid=None):

    # fidelity è weights per le train instances... se multifidelity false setta tutto uguale a 1

    if stratified:
        # for now just oversample higher values in the generation
        pass

    full_data = jnp.load("data/highfidelity/" + filename, allow_pickle=True)

    pores = jnp.asarray(full_data['pores'], dtype=jnp.float32)
    kappas = jnp.asarray(full_data['kappas'], dtype=jnp.float32)

    if multifidelity:
        """high_fid_data = jnp.load("data/highfidelity/" + filename_highfid, allow_pickle=True)
        pores_high = jnp.asarray(high_fid_data['pores'], dtype=jnp.float32)
        kappas_high = jnp.asarray(high_fid_data['kappas'], dtype=jnp.float32)
        base_conductivities_high = jnp.asarray(high_fid_data['conductivity'], dtype=jnp.float32)

        pores = jnp.concatenate([pores, pores_high])
        kappas = jnp.concatenate([kappas, kappas_high])
        base_conductivities = jnp.concatenate([base_conductivities, base_conductivities_high])

        fidelity = jnp.concatenate([fidelity*0.8, ])"""
        # concatenate pores, kappas and so on with highfidelity. Take in account which one are which
        pass


    total_size = len(pores)
    key = key.unwrap() if hasattr(key, "unwrap") else key  # Extract JAX key if it's an nnx RngStream
    indices = jrandom.permutation(key, jnp.arange(total_size))
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    fidelity = jnp.ones(total_size)

    if multifidelity:
        # multiply old fidelity by 0.8 or some value and concatenate 1s relative to the highfidelity ones
        pass

    dataset_train = [pores[train_indices], kappas[train_indices], fidelity[train_indices]]
    dataset_valid = [pores[test_indices], kappas[test_indices], fidelity[test_indices]]


    return dataset_train, dataset_valid, 