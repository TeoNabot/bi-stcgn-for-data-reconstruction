import torch
from torch.utils.data import Dataset
import numpy as np

"""Since the adjacency matrix is static, we do not include it in the dataset and let the model take care of it separately 
The class does not include train/val/test split logic, which must be handled externally."""

import torch
from torch.utils.data import Dataset
import numpy as np

"""
Since the adjacency matrix is static, we do not include it in the dataset and let the model take care of it separately.
The class does not include train/val/test split logic, which must be handled externally.
"""

class STBGNNDataset(Dataset):
    def __init__(
        self,
        X,
        M,
        T_enc, 
        window_size=6,
        mask_ratio=0.5,
        split="train",
        seed=42
    ):
        """
        X: (T, N, F), graph data (Targets: Temp, Humidity, etc.)
        M: (T, N, 1), natural missing sensors per epoch
        T_enc: (T, F_time), Time features (Time covariates: sin_hr, cos_hr, etc.)
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.M = torch.tensor(M, dtype=torch.float32)
        self.T_enc = torch.tensor(T_enc, dtype=torch.float32) # Store time features

        self.W = window_size
        self.mask_ratio = mask_ratio
        self.split = split

        self.T, self.N, self.F = self.X.shape
        self.F_time = self.T_enc.shape[1] # Number of time features

        self.valid_centers = list(
            range(self.W - 1, self.T - self.W)
        )
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.valid_centers)

    def __getitem__(self, idx):
        
        t = self.valid_centers[idx] 

        # --- 1. Slice Windows (Targets) ---
        # Past window (W+1)
        x_past = self.X[t - self.W + 1: t + 2].clone()  # (W+1, N, F_target)
        # Future window (W+1)
        x_future = self.X[t : t + self.W + 1].clone()   # (W+1, N, F_target)
        # Target at center t
        target_vals = self.X[t].clone()  # (N, F_target)

        # --- 2. Slice Windows (Time Covariates) ---
        # <--- NEW: We slice time exactly like X, but we DO NOT mask it later.
        # Since time is global (same for all N nodes), we usually expand it to (N) inside the model,
        # but here we just return the raw time vector for the window.
        
        # Time for past window
        t_past = self.T_enc[t - self.W + 1: t + 2].clone() # (W+1, F_time)
        
        # Time for future window
        t_future = self.T_enc[t : t + self.W + 1].clone()  # (W+1, F_time)
        
        # Time at center t
        t_target = self.T_enc[t].clone() # (F_time)

        # --- 3. Masking Logic (Applied ONLY to Targets X) ---
        loss_mask = torch.zeros(self.N, dtype=torch.float32)

        natural_mask_t = self.M[t].squeeze(-1) # (N,)
        observed_indices = natural_mask_t.nonzero(as_tuple=True)[0].numpy()

        if len(observed_indices) > 0:
            
            n_mask = max(1, int(len(observed_indices) * self.mask_ratio))            
            
            if self.split == "train":
                masked_indices = self.rng.choice(observed_indices, size=n_mask, replace=False)
            else:
                local_rng = np.random.default_rng(self.seed + t)
                masked_indices = local_rng.choice(observed_indices, size=n_mask, replace=False)

            # Apply Mask (Artificial Missingness) to X ONLY
            # Mask current time t in the Past Window (last element)
            x_past[-1, masked_indices, :] = 0.0
            
            # Mask current time t in the Future Window (first element)
            x_future[0, masked_indices, :] = 0.0
            
            # Update Loss Mask
            loss_mask[masked_indices] = 1.0
        
        return {
            "x_past_masked": x_past,      # (W+1, N, F_target) -> Contains 0.0s where masked
            "x_future_masked": x_future,  # (W+1, N, F_target) -> Contains 0.0s where masked
            
            "time_past": t_past,          # (W+1, F_time) -> CLEAN (No masking)
            "time_future": t_future,      # (W+1, F_time) -> CLEAN (No masking)
            
            "target": target_vals,        # (N, F_target)
            "time_target": t_target,      # (F_time)
            
            "loss_mask": loss_mask        # (N,)
        }

    def _is_valid_center(self, t):
        return t >= self.W - 1 and t < self.T - self.W
    
# class STBGNNDataset(Dataset):
#     def __init__(
#         self,
#         X,
#         M,
#         window_size=6,
#         mask_ratio=0.5,
#         split="train",
#         seed=42
#     ):
#         """
#         X: (T, N, F), graph data
#         M: (T, N, 1), natural missing sensors per epoch
#         """
#         self.X = torch.tensor(X, dtype=torch.float32)
#         self.M = torch.tensor(M, dtype=torch.float32)

#         self.W = window_size
#         self.mask_ratio = mask_ratio
#         self.split = split

#         self.T, self.N, self.F = self.X.shape

#         self.valid_centers = list(
#             range(self.W - 1, self.T - self.W)
#         )
#         self.seed = seed
#         self.rng = np.random.default_rng(seed)

#     def __len__(self):
#         return len(self.valid_centers)

#     def __getitem__(self, idx):
        
#         t = self.valid_centers[idx] #map to valid centers automatically ([0]->timestep W-1, [-1]-> timestep T-W)

#         #windows
#         #past window
#         x_past = self.X[t - self.W + 1: t + 2].clone()  # (W+1, N, F)

#         #future window
#         x_future = self.X[t : t + self.W + 1].clone()   # (W+1, N, F)

#         target = self.X[t].clone()  # (N, F). center of the window

#         #what was masked
#         loss_mask = torch.zeros(self.N, dtype=torch.float32)

#         natural_mask_t = self.M[t].squeeze(-1) # (N,)
#         observed_indices = natural_mask_t.nonzero(as_tuple=True)[0].numpy()

#         if len(observed_indices) > 0:
            
#             n_mask = max(1, int(len(observed_indices) * self.mask_ratio))            
            
#             if self.split == "train":
#                 #to get different masks at each epoch, i use the global rng that is stateful, that is, it changes state at each call
#                 #(dataset[0] at epoch 0 will have different masks than dataset[0] at epoch 1)
#                 masked_indices = self.rng.choice(observed_indices, size=n_mask, replace=False)
            
#             else:
#                 # for testing, i want deterministic masks, so i use a local rng seeded by (global_seed + t)
#                 # creating a local new rng ensures that dataset[idx] always has the same masks
#                 local_rng = np.random.default_rng(self.seed + t)
#                 masked_indices = local_rng.choice(observed_indices, size=n_mask, replace=False)

#             # 4. Apply Mask (Artificial Missingness)
#             # The model sees 0.0, but we have the ground truth in 'target'
            
#             # Mask current time t in the Past Window (last element)
#             x_past[-1, masked_indices, :] = 0.0
            
#             # Mask current time t in the Future Window (first element)
#             x_future[0, masked_indices, :] = 0.0
            
#             # Update Loss Mask
#             # We only compute loss on the artificially masked items
#             loss_mask[masked_indices] = 1.0
        
#         return {
#             "x_past_masked": x_past,  #(W+1, N, F)
#             "x_future_masked": x_future,     #(W+1, N, F)
#             "target": target,         #(N, F)
#             "loss_mask": loss_mask    #(N,)
#         }
    
#     def _is_valid_center(self, t):
#         return t >= self.W - 1 and t < self.T - self.W

