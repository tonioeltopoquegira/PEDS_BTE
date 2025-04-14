import os
import seaborn as sns
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax.random as jrandom
from matplotlib import cm
from scipy.stats import gaussian_kde
import numpy as np


from models.model_utils import predict

class DatasetAL:
    def __init__(self, M, N, K, T, dynamic, test_size, converg_criteria_loss, exp_name, model_name, seed):
        full_data = jnp.load(f"data/highfidelity/high_fidelity_2_20000.npz", allow_pickle=True)
        self.key = jrandom.PRNGKey(seed)  # JAX key initialization

        # Load data
        self.pores = jnp.asarray(full_data['pores'], dtype=jnp.float32)
        self.kappas = jnp.asarray(full_data['kappas'], dtype=jnp.float32)

        self.M = M  # Number of new samples per iteration
        self.N = N  # Number of initial samples for training
        self.K = K  # Number of proposed new samples for active learning
        self.last_epoch = 0
        self.T = T  # Epoch increase points
        self.test_size = test_size
        self.dynamic = dynamic
        self.converg_criteria_loss = converg_criteria_loss

        self.exp_name = exp_name
        self.model_name = model_name
        self.iterations = 0

        self.dataset_indices = None  # Training set indices
        self.test_indices = None  # Test set indices

    def initialize(self, rank):
        """Initialize the dataset with N training and N test pairs."""
        self.key, subkey = jrandom.split(self.key)

        all_indices = jnp.arange(self.pores.shape[0])
        selected_indices = jrandom.choice(subkey, all_indices, (self.N+self.test_size,), replace=False)

        # Split into test and initial training set
        self.test_indices = selected_indices[self.N:]
        self.dataset_indices = selected_indices[:self.N]

        # Ensure the test set is not part of the training set
        if rank == 0:
            print(f"AL: Dataset initialized with {len(self.dataset_indices)} training and {len(self.test_indices)} test points")
        
        self.plot_dist()

        return [self.pores[self.dataset_indices], self.kappas[self.dataset_indices]]
        
    def checkupdate(self, epoch, train_loss_history):
        """Check if the dataset should be updated based on the epoch."""
        change = jnp.abs(train_loss_history[epoch-1] - train_loss_history[epoch-20])
        if self.dynamic:
            return change < self.converg_criteria_loss and (epoch - self.last_epoch > 50) 
        else:
            return (epoch + 1) in self.T

    def propose(self, model, comm, rank):
        """Propose M new pores based on uncertainty."""
        # Exclude the test set from the remaining indices
        remaining_indices = jnp.setdiff1d(
            jnp.arange(self.pores.shape[0]),
            jnp.concatenate([self.dataset_indices, self.test_indices])
        )
        
        self.key, subkey = jrandom.split(self.key)
        remaining_indices = jrandom.choice(subkey, remaining_indices, (self.M,), replace=False)

        proposed_pores = self.pores[remaining_indices]
        if rank == 0:
            print(f'Proposed {len(proposed_pores)} new samples.')

        # Use the model to predict the mean of the new proposed samples
        kappa_mean, _ = predict(model, proposed_pores, training=False)

        comm.Barrier()

        # Gather the kappa mean predictions from all ranks
        kappa_pred_all = comm.allgather(kappa_mean)
        kappa_pred_stacked = jnp.stack(kappa_pred_all)

        # Calculate variance (uncertainty) across the ranks
        kappa_pred_var = jnp.var(kappa_pred_stacked, axis=0)

        # Select K points with the highest uncertainty (variance)
        top_indices = jnp.argsort(-kappa_pred_var)[:self.K]
        if rank == 0:
            print(f'Taken {len(top_indices)} samples with highest uncertainty:\n {kappa_pred_var[top_indices[:5]]}')

        return remaining_indices[top_indices]

    def sample(self, model, comm, rank, epoch):
        """Expand the dataset with proposed points."""
        new_indices = self.propose(model, comm, rank)
        self.dataset_indices = jnp.concatenate([self.dataset_indices, new_indices])

        self.iterations+=1
        self.last_epoch = epoch

        if rank == 0:
            print(f"{self.iterations}) AL: Dataset has now {len(self.dataset_indices)} points")
            self.plot_dist()

        return [self.pores[self.dataset_indices], self.kappas[self.dataset_indices]]

    def plot_dist(self):
        """ Plots the train distribution using KDE (Kernel Density Estimation) """
        os.makedirs(f"experiments/{self.exp_name}/figures/al/", exist_ok=True)
        plt.clf()

        # Normalize iteration number to [0, 1] for colormap
        max_iter = 20
        norm_iter = min(self.iterations / max_iter, 1.0)
        cmap = cm.get_cmap("cividis")
        color = cmap(norm_iter)

        data = self.kappas[self.dataset_indices]

        # Compute KDE manually to extract max
        kde = gaussian_kde(data)
        x_vals = np.linspace(0, 175, 1000)
        density_vals = kde(x_vals)

        self.y_ax = 0.06  # Save max density for consistent y-axis

        # Plot using seaborn (for nice visuals)
        sns.kdeplot(data, fill=True, color="blue")
        
        plt.title(f"Training Distribution at {self.iterations} AL iteration")
        plt.xlabel("Pores")
        plt.ylabel("Density")
        plt.xlim(0, 175)
        plt.ylim(0, self.y_ax)

        plt.savefig(f'experiments/{self.exp_name}/figures/al/{self.model_name}_{self.iterations}.png')


    def get_test_set(self):
        """Return the reserved test set."""
        return [self.pores[self.test_indices], self.kappas[self.test_indices]]
