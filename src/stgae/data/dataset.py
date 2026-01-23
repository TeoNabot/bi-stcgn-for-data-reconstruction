import torch
from torch.utils.data import Dataset
import numpy as np

"""since the adjacency matrix is static, i do not include it in the dataset and let the model take care of it separately 
the class does not include train/val/test split logic, which must be handled externally."""


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
        X: (T, N, F), graph data
        M: (T, N, 1), natural missing sensors per epoch
        T_enc: (T, F_time), Time features
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.M = torch.tensor(M, dtype=torch.float32)
        self.T_enc = torch.tensor(T_enc, dtype=torch.float32)

        self.W = window_size
        self.mask_ratio = mask_ratio
        self.split = split

        self.T, self.N, self.F = self.X.shape
        self.F_time = self.T_enc.shape[1]

        self.valid_centers = list(
            range(self.W - 1, self.T - self.W)
        )
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.valid_centers)

    def __getitem__(self, idx):
        
        t = self.valid_centers[idx] 

        x_past = self.X[t - self.W + 1: t + 2].clone()  # (W+1, N, F_target)
        x_future = self.X[t : t + self.W + 1].clone()   # (W+1, N, F_target)
        target_vals = self.X[t].clone()  # (N, F_target)
        
        t_past = self.T_enc[t - self.W + 1: t + 2].clone() # (W+1, F_time)
        t_future = self.T_enc[t : t + self.W + 1].clone()  # (W+1, F_time)
        t_target = self.T_enc[t].clone() # (F_time)

        #masking
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

            # apply mask
            x_past[-1, masked_indices, :] = 0.0
            x_future[0, masked_indices, :] = 0.0
            
            #save masked indices
            loss_mask[masked_indices] = 1.0
        
        return {
            "x_past_masked": x_past,      # (W+1, N, F_target). past window with a set of sensors features at time t masked (set to zero).
            "x_future_masked": x_future,  # (W+1, N, F_target)  future window with a set of sensors features at time t masked (set to zero).
            
            "time_past": t_past,          # (W+1, F_time) -> past window time features not masked
            "time_future": t_future,      # (W+1, F_time). past window time features not masked
            
            "target": target_vals,        # (N, F_target)
            "time_target": t_target,      # (F_time)
            
            "loss_mask": loss_mask        # (N,)
        }

    def _is_valid_center(self, t):
        return t >= self.W - 1 and t < self.T - self.W
    