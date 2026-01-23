from pathlib import Path
from stgae.config.load_config import load_config
import pandas as pd
import numpy as np
import torch 

columns = ['date', 'time', 'epoch', 'moteid', 'temperature', 'humidity', 'light', 'voltage']

def filter_data_errors(df):
    #as stated in main notebook
    constraints = {
        'voltage': (2.0, 3.2),     
        'temperature': (8.0, 38.0), 
        'humidity': (0.1, 99.9),   
        'light': (0.0, 2000.0)    
    }

    mask = pd.Series(True, index=df.index)

    for col, (min_val, max_val) in constraints.items():
        if col in df.columns:
            #keep if inside range
            col_mask = (df[col] >= min_val) & (df[col] <= max_val)
            mask = mask & col_mask

    df_clean = df[mask].copy()
    
    return df_clean

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
    calculates the normalized adjacency matrix for a graph, excluding the given sensors
    based on knn approach (include edge v, w if v is knn of w or w is knn of v)
    returns normalized matrix (neighbors add to 1)
    """
    N_orig = dist_matrix.shape[0]
    
    #indeces that are not in exclude
    keep_indices = [i for i in range(N_orig) if i not in exclude]
    
    #keep valid indices
    D = dist_matrix[np.ix_(keep_indices, keep_indices)].copy()
    
    N = D.shape[0]


    #find k nearest neighbors
    np.fill_diagonal(D, np.inf) #exclude self nodes
    k_actual = min(k, N - 1)
    knn_idx = np.argsort(D, axis=1)[:, :k_actual]

    #determine which edges to include
    A = np.zeros((N, N))

    for i in range(N):
        for j in knn_idx[i]:
            A[i, j] = 1

    #i include the edge if i is nighbor of v or viceversa -> symettric
    A = np.maximum(A, A.T)

    #i add weights
    eps = 1e-6
    W = np.zeros_like(A)

    for i in range(N):
        for j in range(N):
            if A[i, j] == 1:
                W[i, j] = 1.0 / (D[i, j] + eps)

    #normalize
    #sym
    A_sym = np.maximum(W, W.T)
    
    #add self loops
    max_weights = A_sym.max(axis=1) 
    # Ensure isolated nodes have self-loop weight of 1.0 to avoid division by zero
    if sum(max_weights == 0) > 0 : print('isolated nodes')
    max_weights[max_weights == 0] = 1.0
    A_self = A_sym + np.diag(max_weights)
    
    # normalization per node
    row_sum = A_self.sum(axis=1)
    
    D_inv = np.diag(1.0 / row_sum)
    
    # A_norm = D^-1 * A
    A_norm = D_inv @ A_self
    return torch.tensor(A_norm, dtype=torch.float32)

def build_tensors(df, epochs, sensors, feature_cols, time_cols):
    T = len(epochs)
    N = len(sensors)
    F = len(feature_cols)

    X = np.zeros((T, N, F), dtype=np.float32)
    M = np.zeros((T, N), dtype=np.float32)

    epoch_cat = pd.Categorical(df['epoch'], categories=epochs, ordered=True)
    mote_cat = pd.Categorical(df['moteid'], categories=sensors, ordered=True)

    t_idx = epoch_cat.codes
    n_idx = mote_cat.codes
    
    valid = (t_idx >= 0) & (n_idx >= 0)

    if np.any(valid):
        feats = df.loc[valid, feature_cols].to_numpy(dtype=np.float32)
        X[t_idx[valid], n_idx[valid]] = feats
        M[t_idx[valid], n_idx[valid]] = 1.0

    unique_time_df = df.drop_duplicates(subset=['epoch']).set_index('epoch')
    T_enc = unique_time_df[time_cols].reindex(epochs).to_numpy(dtype=np.float32)

    return X, M, T_enc

def get_columns():
    return columns

class MaskedStandardScaler:
    """
    standard Scaler that ignores masked valus
    """
    def __init__(self, axis=(0, 1), epsilon=1e-5):
        self.mean = None
        self.std = None
        self.axis = axis
        self.epsilon = epsilon

    def fit(self, X, M):
        if torch.is_tensor(X):
            X = X.numpy()
            M = M.numpy()

        #expand mask from (T, N) to (T, N, F) to match X for calculation
        if M.ndim == 2 and X.ndim == 3:
            M = np.expand_dims(M, axis=-1)
            #broadcast mask across features
            M = np.tile(M, (1, 1, X.shape[-1]))

        X_masked = np.ma.masked_array(X, mask=(M == 0))

        self.mean = X_masked.mean(axis=self.axis)
        self.std = X_masked.std(axis=self.axis)
        
        #eeplace 0 std with 1 to avoid NaNs
        self.std[self.std < self.epsilon] = 1.0
        
        return self

    def transform(self, X, M=None, remask_zeros=True):
        """
        normalizes X. 
        if remask_zeros is True, values where M=0 are forced back to 0.
        """
        is_tensor = torch.is_tensor(X)
        if is_tensor:
            X_np = X.numpy()
        else:
            X_np = X

        # reshape mean/std for broadcasting
        mean_reshaped = np.expand_dims(np.expand_dims(self.mean, 0), 0)
        std_reshaped = np.expand_dims(np.expand_dims(self.std, 0), 0)

        X_norm = (X_np - mean_reshaped) / std_reshaped

        if remask_zeros and M is not None:
            if torch.is_tensor(M): M = M.numpy()
            if M.ndim == 2: M = np.expand_dims(M, -1)
            
            # missing values to 0
            X_norm = X_norm * M 
        
        #return in same format as input
        if is_tensor:
            return torch.FloatTensor(X_norm)
        return X_norm

    def inverse_transform(self, X_norm):
        """
        reverts normalization (to go back to iriginal scale).
        """
        is_tensor = torch.is_tensor(X_norm)
        if is_tensor:
            X_np = X_norm.cpu().numpy()
        else:
            X_np = X_norm

        mean_reshaped = np.expand_dims(np.expand_dims(self.mean, 0), 0)
        std_reshaped = np.expand_dims(np.expand_dims(self.std, 0), 0)

        X_orig = (X_np * std_reshaped) + mean_reshaped

        if is_tensor:
            return torch.FloatTensor(X_orig).to(X_norm.device)
        return X_orig
