import jax.numpy as jnp
import jax.random as jrandom

from models.model_utils import predict

class DatasetAL:
    def __init__(self, filename, M, N, K, T, seed):
        full_data = jnp.load(f"data/highfidelity/{filename}", allow_pickle=True)
        self.key = jrandom.PRNGKey(seed)  # JAX key initialization

        # Load data
        self.pores = jnp.asarray(full_data['pores'], dtype=jnp.float32)
        self.kappas = jnp.asarray(full_data['kappas'], dtype=jnp.float32)

        self.M = M  # Number of new samples per iteration
        self.N = N  # Number of initial and test samples
        self.K= K
        self.T = T

        self.dataset_indices = None  # Training set indices
        self.test_indices = None  # Test set indices

    def initialize(self):
        """Initialize the dataset with N training and N test pairs."""
        self.key, subkey = jrandom.split(self.key)

        all_indices = jnp.arange(self.pores.shape[0])
        selected_indices = jrandom.choice(subkey, all_indices, (2 * self.N,), replace=False)

        # Split into test and initial training set
        self.test_indices = selected_indices[:self.N]
        self.dataset_indices = selected_indices[self.N:]
        print(f"AL: Dataset initialized with {len(self.dataset_indices)} points")

        return [self.pores[self.dataset_indices], self.kappas[self.dataset_indices]]
        
    def checkupdate(self, epoch):
        return (epoch+1)% self.T == 0


    def propose(self, model):
        """Propose M new pores based on uncertainty."""
        remaining_indices = jnp.setdiff1d(jnp.arange(self.pores.shape[0]), jnp.concatenate([self.dataset_indices, self.test_indices]))
        proposed_pores = self.pores[remaining_indices]

        print(type(model))

        kappa_mean, kappa_var = predict(model, proposed_pores, training=False)

        # Select M/3 points with highest uncertainty
        top_indices = jnp.argsort(-kappa_var)[: self.K]

        return remaining_indices[top_indices]

    def sample(self, model, iteration=0):
        """Expand the dataset with proposed points."""
        new_indices = self.propose(model)
        self.dataset_indices = jnp.concatenate([self.dataset_indices, new_indices])

        print(f"AL: Dataset has now {len(self.dataset_indices)} points")

        return [self.pores[self.dataset_indices], self.kappas[self.dataset_indices]]

    def get_test_set(self):
        """Return the reserved test set."""
        return [self.pores[self.test_indices], self.kappas[self.test_indices]]