import torch
from torch.utils.data import Dataset
import numpy as np

class STBGNNDataset(Dataset):
    def __init__(
        self,
        X,
        M,
        window_size=12,
        mask_ratio=0.2,
        split="train",
        seed=42
    ):
        """
        X: (T, N, F)
        M: (T, N, 1)
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
        if(not(self._is_valid_center(idx))):
            raise IndexError("Index out of valid center range.")
        
        t = idx

        #windows
        #past windows
        x_past = self.X[t - self.W + 1 : t + 1]      # (W, N, F)
        m_past = self.M[t - self.W + 1 : t + 1]      # (W, N, 1)

        #future windows
        x_future = self.X[t + 1 : t + self.W + 1]   # (W, N, F)
        m_future = self.M[t + 1 : t + self.W + 1]   # (W, N, 1)

        # Reverse future for backward branch
        x_bwd = torch.flip(x_future, dims=[0])
        m_bwd = torch.flip(m_future, dims=[0])

        #apply artificial mask for training 
        if self.split == "train":
            mask_train = self._apply_sensor_mask(m_past)
        else:
            mask_train = m_past.clone()

        #masked tensors for training, otherwise original
        x_fwd = x_past * mask_train
        x_bwd = x_bwd * mask_train

        # Loss mask: artificially masked & originally observed
        mask_loss = (m_past - mask_train).clamp(min=0)

        target = self.X[t]  # (N, F)

        return {
            "x_fwd": x_fwd,
            "x_bwd": x_bwd,
            "mask_obs_past": m_past,
            "mask_obs_future": m_future,
            "mask_train": mask_train,
            "mask_loss": mask_loss,
            "target": target
        }
    
    def _is_valid_center(self, t):
        return t >= self.W - 1 and t < self.T - self.W
    
    def _apply_sensor_mask(self, m):
        """
        m: (W, N, 1)
        """
        mask = m.clone()

        # sensors observed at center time
        observed = m[-1, :, 0].nonzero(as_tuple=True)[0]

        n_mask = int(len(observed) * self.mask_ratio)

        if n_mask == 0:
            return mask

        masked_sensors = self.rng.choice(
            observed.cpu().numpy(),
            size=n_mask,
            replace=False
        )

        mask[:, masked_sensors, :] = 0.0
        return mask

