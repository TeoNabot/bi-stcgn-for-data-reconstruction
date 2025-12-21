from pathlib import Path
from stgae.config.load_config import load_config
import pandas as pd
import numpy as np
import torch 

columns = ['date', 'time', 'epoch', 'moteid', 'temperature', 'humidity', 'light', 'voltage']

def preprocess():
    paths = load_config()['paths']    
    raw_data_path = Path(paths['raw_data'])

def calculate_distances(coordinates: pd.DataFrame) -> np.array:
    #receives a df with columns ['moteid', 'x', 'y']
    #returns a matrix of distances

    #sort by moteid to ensure correct order
    coordinates = coordinates.sort_values('moteid')

    coords = coordinates[['x', 'y']].to_numpy()
    dist_matrix = np.linalg.norm(coords[:, np.newaxis] - coords[np.newaxis, :], axis=-1)
    return dist_matrix    

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

import numpy as np
import torch

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
    T = len(epochs)
    N = len(sensors)
    F = len(feature_cols)

    X = np.zeros((T, N, F), dtype=np.float32)
    M = np.zeros((T, N, 1), dtype=np.float32)

    for _, row in df.iterrows():
        t = row["epoch"]
        n = row["moteid"]

        X[t, n] = row[feature_cols].values
        M[t, n] = 1.0

    return X, M

def calculate_anomalies():
    pass

def add_anomalies():
    pass

def get_columns():
    return columns

def create_graph():
    pass
