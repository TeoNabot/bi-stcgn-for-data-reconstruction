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

def calculate_adjacency_matrix(dist_matrix, k):
    D = dist_matrix.copy()
    np.fill_diagonal(D, np.inf)

    #find k-nearest neighbors
    k = 5
    knn_idx = np.argsort(D, axis=1)[:, :k]

    #build adjacency matrix
    N = D.shape[0]
    A = np.zeros((N, N))

    for i in range(N):
        for j in knn_idx[i]:
            A[i, j] = 1

    #ensure symetric
    A = np.maximum(A, A.T)

    #add weights edges
    eps = 1e-6
    W = np.zeros_like(A)

    for i in range(N):
        for j in range(N):
            if A[i, j] == 1:
                W[i, j] = 1.0 / (D[i, j] + eps)


    #normalize adjacency for GCN

    #self loops
    W_tilde = W + np.eye(N)
    #symetric normalization
    deg = W_tilde.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(deg))
    A_norm = D_inv_sqrt @ W_tilde @ D_inv_sqrt

    return torch.tensor(A_norm, dtype=torch.float32)


def calculate_anomalies():
    pass

def add_anomalies():
    pass

def get_columns():
    return columns

def create_graph():
    pass
