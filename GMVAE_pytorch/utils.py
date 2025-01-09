import numpy as np
from scipy.spatial.distance import pdist
from scipy import stats
import torch
from torch.utils.data import Dataset

class TimeLaggedDataset(Dataset):
    def __init__(self, data, time_lag):
        self.data = data
        self.time_lag = time_lag

    def __len__(self):
        return len(self.data) - self.time_lag

    def __getitem__(self, idx):
        x_input = self.data[idx]
        x_output = self.data[idx + self.time_lag]
        return torch.tensor(x_input, dtype=torch.float32), torch.tensor(x_output, dtype=torch.float32)

def analyze_gmvae_clustering(qy):
    """
    Analyze GMVAE clustering results.

    Parameters:
    qy (numpy.ndarray): Array of cluster probabilities or predictions.

    Returns:
    dict: A dictionary containing cluster data with percentages and frame indices.
    """
    # Get the predicted cluster for each data point
    y_pred = np.argmax(qy, axis=1)
    
    # Get unique cluster indices
    y_unq = np.unique(y_pred)
    
    # Normalize cluster indices
    for i in range(len(y_unq)):
        y_pred[np.where(y_pred == y_unq[i])] = np.arange(0, len(y_unq))[i]

    # Calculate cluster statistics
    _, counts = np.unique(y_pred, return_counts=True)
    percentages = dict(zip(y_unq, np.round(100 * counts / len(y_pred), 2)))

    # Print summary of clustering results
    print("GMVAE Clustering Results:\n")
    print("Overview:")
    print("  Number of Clusters: ", len(y_unq))
    print("  Cluster Indices: ", y_unq)
    print("")
    print("Cluster Population:")
    for key, value in percentages.items():
        print(f"  {key}: {value:.2f} %")
        
def print_traj_properties(traj):

    # Print trajectory information from mdtraj object
    print("Trajectory Information:")
    print(f"  Number of atoms: {traj.topology.n_atoms}")
    print(f"  Number of residues: {traj.topology.n_residues}")
    print(f"  Number of chains: {traj.topology.n_chains}")
    print(f"  Number of frames: {traj.n_frames}")
    print(f"  Time between frames (ps): {traj.timestep}")
    print(f"  Total time (ps): {traj.timestep * traj.n_frames}")
    print(f"  Unit cell dimensions: {traj.unitcell_lengths}\n")

def pairwise_distances(traj, cutoff=None, batch_size=1000):
    """
    Compute pairwise distances for a trajectory, keeping pairs if they are within the
    cutoff distance at least once during the entire trajectory.

    Parameters:
    - traj (np.array): Trajectory array of shape [frames, atoms, xyz].
    - cutoff (float): Distance cutoff; pairs within this distance at any point are included.
    - batch_size (int): Number of frames to process in each batch.

    Returns:
    - distances (np.array): Array of distances with shape [frames, n_interactions].
    - indices (np.array): Array of pairs of atom indices that were included, shape [n_interactions, 2].
    """
    num_frames, num_atoms, _ = traj.shape
    distances = []

    # Indices of the upper triangle pairs
    iu = np.triu_indices(num_atoms, k=1)

    # Initialize valid_pairs as an empty mask; will be updated across all batches
    if cutoff is not None:
        valid_pairs = np.zeros(len(iu[0]), dtype=bool)

    # Process by batches
    for start in range(0, num_frames, batch_size):
        end = start + batch_size
        batch_traj = traj[start:end]
        
        # Calculate distances for each frame in the batch
        batch_distances = np.array([pdist(batch_traj[i]) for i in range(len(batch_traj))])
        
        if cutoff is not None:
            # Update valid_pairs to include distances below cutoff in any frame
            valid_pairs |= np.any(batch_distances < cutoff, axis=0)
        
        distances.append(batch_distances)

    if cutoff is not None:
        # Filter indices using the valid_pairs mask
        iu = np.vstack(iu).T[valid_pairs]  # Shape [n_interactions, 2]
        distances = [dist[:, valid_pairs] for dist in distances]

    # Convert list to array for final output, adjusting dimensions for convenience
    if distances:
        distances = np.concatenate(distances, axis=0)
    print("Distances shape: ", distances.shape) 
    return distances, iu

def kurtosis_filtering(distances, indices, threshold=0.03):
    # distances [frames, dist]; indices [dist, (i, j)]
    # Calculate kurtosis for each pair across all frames
    kurtosis_values = np.apply_along_axis(stats.kurtosis, 0, distances)

    # Find pairs with kurtosis greater than the threshold
    valid_pairs = kurtosis_values > threshold

    # Filter distances and indices based on the kurtosis condition
    filtered_distances = distances[:, valid_pairs]
    filtered_indices = indices[valid_pairs]
    # filtered_distances [frames, kurt_dist]; filtered_distances [dist, kurt(i,j)]; kurtosis_values: score
    print("Distances after kurtosis: ", filtered_distances.shape)
    return filtered_distances, filtered_indices, kurtosis_values[valid_pairs]
