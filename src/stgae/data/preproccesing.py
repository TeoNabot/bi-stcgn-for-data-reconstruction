from pathlib import Path
from stgae.config.load_config import load_config
import pandas as pd
import numpy as np
import torch 

columns = ['date', 'time', 'epoch', 'moteid', 'temperature', 'humidity', 'light', 'voltage']

def calculate_distances(coordinates: pd.DataFrame) -> np.array:
    #receives a df with columns ['moteid', 'x', 'y']
    #returns a matrix of distances

    #sort by moteid to ensure correct order
    coordinates = coordinates.sort_values('moteid')

    coords = coordinates[['x', 'y']].to_numpy()
    dist_matrix = np.linalg.norm(coords[:, np.newaxis] - coords[np.newaxis, :], axis=-1)
    return dist_matrix    


def calculate_adjacency_matrix(dist_matrix, k, exclude):
    """
    Calculates the normalized adjacency matrix for a graph, 
    physically removing the excluded sensors from the matrix.
    
    Output shape: (N_new, N_new) where N_new = N_original - len(exclude)
    """
    N_orig = dist_matrix.shape[0]
    
    # 1. Identify indices to keep
    # Create a list of indices that are NOT in the exclude list
    keep_indices = [i for i in range(N_orig) if i not in exclude]
    
    # 2. Slice the original matrix to keep only valid sensors
    # np.ix_ allows us to slice both rows and columns simultaneously
    D = dist_matrix[np.ix_(keep_indices, keep_indices)].copy()
    
    # The new size of the matrix
    N = D.shape[0]

    # 3. Standard k-NN Logic (Same as before, but on the reduced matrix)
    np.fill_diagonal(D, np.inf)

    # Find k-nearest neighbors
    # Note: If k >= N, we clip it to N-1 to avoid errors
    k_actual = min(k, N - 1)
    knn_idx = np.argsort(D, axis=1)[:, :k_actual]

    # Build binary adjacency matrix
    A = np.zeros((N, N))

    for i in range(N):
        for j in knn_idx[i]:
            A[i, j] = 1

    # Ensure symmetric
    A = np.maximum(A, A.T)

    # Add weighted edges
    eps = 1e-6
    W = np.zeros_like(A)

    for i in range(N):
        for j in range(N):
            if A[i, j] == 1:
                # D contains the distances of the kept sensors
                W[i, j] = 1.0 / (D[i, j] + eps)

    # 4. Normalize adjacency for GCN
    
    # Self loops
    W_tilde = W + np.eye(N)
    
    # Symmetric normalization: D_inv_sqrt @ W_tilde @ D_inv_sqrt
    deg = W_tilde.sum(axis=1)
    # Avoid division by zero if a node is disconnected (though unlikely with self-loops)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(deg + eps)) 
    
    A_norm = D_inv_sqrt @ W_tilde @ D_inv_sqrt

    return torch.tensor(A_norm, dtype=torch.float32)

def build_tensors(df, epochs, sensors, feature_cols):
    """
    Vectorized build of tensors from a long dataframe.

    Parameters
    - df: pandas DataFrame containing at least columns 'epoch' and 'moteid' plus feature_cols
    - epochs: sequence of epoch identifiers (order defines time indices 0..T-1)
    - sensors: sequence of sensor identifiers (order defines node indices 0..N-1)
    - feature_cols: list of feature column names

    Assumes each (epoch, moteid) pair appears at most once. Rows with epoch/moteid
    values not present in the provided `epochs`/`sensors` lists are ignored.
    """

    T = len(epochs)
    N = len(sensors)
    F = len(feature_cols)

    X = np.zeros((T, N, F), dtype=np.float32)
    M = np.zeros((T, N, 1), dtype=np.float32)

    # Map dataframe epoch/moteid values to index positions defined by the provided lists.
    # Using pandas.Categorical is fast and preserves the order of the provided categories.
    epoch_cat = pd.Categorical(df['epoch'], categories=epochs, ordered=True)
    mote_cat = pd.Categorical(df['moteid'], categories=sensors, ordered=True)

    t_idx = epoch_cat.codes  # -1 where epoch not in categories
    n_idx = mote_cat.codes   # -1 where moteid not in categories

    valid = (t_idx >= 0) & (n_idx >= 0)
    if not np.all(valid):
        # silently ignore rows that don't belong to the provided epochs/sensors
        t_idx = t_idx[valid]
        n_idx = n_idx[valid]
        feats = df.loc[valid, feature_cols].to_numpy(dtype=np.float32)
    else:
        feats = df[feature_cols].to_numpy(dtype=np.float32)

    # Bulk assign feature values and mask
    X[t_idx, n_idx] = feats
    M[t_idx, n_idx, 0] = 1.0

    return X, M

def get_columns():
    return columns

class StandardScaler:
    """
    Standardize the input data by removing the mean and scaling to unit variance.
    Assumes data shape: (Samples, Nodes, Features)
    """
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def fit(self, data):
        """
        Compute the mean and std to be used for later scaling.
        data: numpy array (T, N, F)
        """
        # We aggregate over Time (0) and Nodes (1) to get stats per Feature (F)
        # If nodes have vastly different behaviors, you might only aggregate over Time (0).
        # Usually for traffic, aggregating over (0, 1) is fine (shared characteristics).
        self.mean = np.mean(data, axis=(0, 1), keepdims=True) # Shape (1, 1, F)
        self.std = np.std(data, axis=(0, 1), keepdims=True)   # Shape (1, 1, F)
        
        # Prevent division by zero for constant features
        self.std[self.std < 1e-5] = 1.0 

    def transform(self, data):
        """
        Perform standardization by centering and scaling.
        """
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        """
        Scale back the data to the original representation.
        Supports both numpy and torch tensors.
        """
        if torch.is_tensor(data):
            # Move mean/std to the same device as data
            mean_t = torch.as_tensor(self.mean, device=data.device, dtype=data.dtype)
            std_t = torch.as_tensor(self.std, device=data.device, dtype=data.dtype)
            return (data * std_t) + mean_t
        else:
            return (data * self.std) + self.mean
        
        # def calculate_adjacency_matrix(dist_matrix, k, exclude):
#     D = dist_matrix.copy()
#     np.fill_diagonal(D, np.inf)
#     N = D.shape[0]

#     #to exclude sensors set distances to infinity
#     for sensor in exclude:
#         for i in range(N):
#             D[i, sensor] = np.inf
#             D[sensor, i] = np.inf

#     #find k-nearest neighbors
#     knn_idx = np.argsort(D, axis=1)[:, :k]

#     #build adjacency matrix
#     A = np.zeros((N, N))

#     for i in range(N):
#         for j in knn_idx[i]:
#             A[i, j] = 1

#     #ensure symetric
#     A = np.maximum(A, A.T)

#     #add weights edges
#     eps = 1e-6
#     W = np.zeros_like(A)

#     for i in range(N):
#         for j in range(N):
#             if A[i, j] == 1:
#                 W[i, j] = 1.0 / (D[i, j] + eps)


#     #normalize adjacency for GCN

#     #self loops
#     W_tilde = W + np.eye(N)
#     #symetric normalization
#     deg = W_tilde.sum(axis=1)
#     D_inv_sqrt = np.diag(1.0 / np.sqrt(deg))
#     A_norm = D_inv_sqrt @ W_tilde @ D_inv_sqrt

#     return torch.tensor(A_norm, dtype=torch.float32)