import torch
from torch.utils.data import Dataset
import numpy as np

"""Since the adjacency matrix is static, we do not include it in the dataset and let the model take care of it separately 
The class does not include train/val/test split logic, which must be handled externally."""

class STBGNNDataset(Dataset):
    def __init__(
        self,
        X,
        M,
        window_size=6,
        mask_ratio=0.5,
        split="train",
        seed=42
    ):
        """
        X: (T, N, F), graph data
        M: (T, N, 1), natural missing sensors per epoch
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.M = torch.tensor(M, dtype=torch.float32)

        self.W = window_size
        self.mask_ratio = mask_ratio
        self.split = split

        self.T, self.N, self.F = self.X.shape

        self.valid_centers = list(
            range(self.W - 1, self.T - self.W)
        )

        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.valid_centers)

    def __getitem__(self, idx):
        
        t = self.valid_centers[idx] #map to valid centers automatically ([0]->timestep W-1, [-1]-> timestep T-W)

        #windows
        #past window
        x_past = self.X[t - self.W + 1: t + 2].clone()  # (W+1, N, F)

        #future window
        x_future = self.X[t : t + self.W + 1].clone()   # (W+1, N, F)

        target = self.X[t].clone()  # (N, F). center of the window

        #what was masked
        loss_mask = torch.zeros(self.N, dtype=torch.float32)

        natural_mask_t = self.M[t].squeeze(-1) # (N,)
        observed_indices = natural_mask_t.nonzero(as_tuple=True)[0].numpy()

        if len(observed_indices) > 0:
            
            n_mask = max(1, int(len(observed_indices) * self.mask_ratio))            
            
            if self.split == "train":
                #to get different masks at each epoch, i use the global rng that is stateful, that is, it changes state at each call
                #(dataset[0] at epoch 0 will have different masks than dataset[0] at epoch 1)
                masked_indices = self.rng.choice(observed_indices, size=n_mask, replace=False)
            
            else:
                # for testing, i want deterministic masks, so i use a local rng seeded by (global_seed + t)
                # creating a local new rng ensures that dataset[idx] always has the same masks
                local_rng = np.random.default_rng(self.seed + t)
                masked_indices = local_rng.choice(observed_indices, size=n_mask, replace=False)

            # 4. Apply Mask (Artificial Missingness)
            # The model sees 0.0, but we have the ground truth in 'target'
            
            # Mask current time t in the Past Window (last element)
            x_past[-1, masked_indices, :] = 0.0
            
            # Mask current time t in the Future Window (first element)
            x_future[0, masked_indices, :] = 0.0
            
            # Update Loss Mask
            # We only compute loss on the artificially masked items
            loss_mask[masked_indices] = 1.0
        
        return {
            "x_past_masked": x_past,  #(W+1, N, F)
            "x_future_masked": x_future,     #(W+1, N, F)
            "target": target,         #(N, F)
            "loss_mask": loss_mask    #(N,)
        }
    
    def _is_valid_center(self, t):
        return t >= self.W - 1 and t < self.T - self.W
