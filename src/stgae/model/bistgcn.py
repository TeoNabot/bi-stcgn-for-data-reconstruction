"""
The model should learn at a center time t, 
and given partial observations of sensors in a past and future window,
to reconstruct the missing sensor values at time t
"""

"""
Since the graph is static and the main difficulties are related to time and masking,
I used standard PyTorch and not PyG
"""

"""
Assumed:
adj matrix is precomputed and normalized
masking handled in dataset
input tensors have shape (B, W, N, F)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
B: batch size
w: window length (W - center - W)
N: number of nodes
F: number of feautures"""

class BiSTGCN(nn.Module):
    def __init__(self, in_features, hidden_dim, adj, kernel_size=3):
        super().__init__()

        #precesses past window + t 
        #when training, data at t must be artificially masked
        self.fwd_block = STGCNBlock(in_features=in_features, hidden_dim=hidden_dim, adj=adj, kernel_size=kernel_size)

        #processes future window
        self.bwd_block = STGCNBlock(in_features=in_features, hidden_dim=hidden_dim, adj=adj, kernel_size=kernel_size)

        # fuse forward and backward hidden representations and project back to input feature size
        self.fuse = nn.Linear(2 * hidden_dim, in_features)

        # output activation
        self.output_activation = nn.Identity()

    def forward(self, x_past, x_future):
        """
        x_past: (B, W+1, N, F) (past window, including data at time t, masked if training)
        x_future: (B, W, N, F) (future window)
        x: (B, N, F) (X at time t, current target)
        masked_x: (B, N, F) (X at time t, used at training to predict curent target)
        """
        B, W, N, F = x_past.shape
        
        # forward pass for past window + t
        past_out = self.fwd_block(x_past)   # (B, W+1, N, H)

        #extract the state at time t (the last element)
        current_state_out = past_out[:, -1, :, :] # (B, N, H)

        # bacward for future window
        x_rev = x_future.flip(dims=[1])
        future_out = self.bwd_block(x_rev)

        #extract state t+1, last element of the flipped future out
        future_context = future_out[:, -1, :, :] # (B, N, H)

        # Concatenate the estimation derived from (Past+Spatial_t) with (Future)
        combined = torch.cat([current_state_out, future_context], dim=-1) # (B, N, 2H)

        # project to original feature size F
        out_center = self.fuse(combined) # (B, N, F)

        return self.output_activation(out_center)    
        

class GraphConv(nn.Module):
    def __init__(self, in_features, out_features, adj):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        #adj not trainable, moves with model
        self.register_buffer("adj", adj)

        #same transformation for all nodes and timesteps (for mat mul, shape FxH)
        self.weight = nn.Parameter(
            torch.empty(in_features, out_features)
        )

        #weight initialization
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        """
        x size: (B, W, N, F)
        """
        # 1. Linear projection: (B, W, N, F) -> (B, W, N, H)
        x = torch.matmul(x, self.weight)

        # 2. Message Passing
        # adj: (N, N) -> "nm" (Target Node n, Neighbor Node m)
        # x: (B, W, N, H) -> "bwmh" (Batch, Window, Neighbor Node m, Hidden h)
        # result: (B, W, N, H) -> "bwnh" (aggregating over m)
        
        x = torch.einsum("nm,bwmh->bwnh", self.adj, x) 

        return x

class STGCNBlock(nn.Module):
    def __init__(self, in_features, hidden_dim, adj, kernel_size=3):
        super().__init__()

        self.spatial = GraphConv(
            in_features=in_features,
            out_features=hidden_dim,
            adj=adj
        )

        self.temporal = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2 #to mantain window size
        )

        self.activation = nn.ReLU()

    def forward(self, x):
        """
        x: (B, W, N, F)
        """
        B, W, N, _ = x.shape

        # spatial (graph conv)
        x = self.spatial(x)        # (B, W, N, H) = GraphConv(B, W, N, F)
        x = self.activation(x)

        # temporal 1d conv
        
        x = x.permute(0, 2, 3, 1)  # (B, N, H, W). pytorch expects time as the last dimension in Conv1d
        x = x.reshape(B * N, -1, W)  # (B*N, H, W), flatten batch, to apply to 

        x = self.temporal(x)       # (B*N, H, W)
        x = self.activation(x)

        # reshape back
        x = x.reshape(B, N, -1, W)
        x = x.permute(0, 3, 1, 2)  # (B, W, N, H)

        return x
