import torch
import torch.nn as nn
import torch.nn.functional as F

"""
B: batch size
w: window length (W - center - W)
N: number of nodes
F: number of feautures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class BiSTGCN(nn.Module):
    def __init__(self, target_dim, time_dim, hidden_dim, adj, kernel_size=3):
        super().__init__()

        input_dim = target_dim + time_dim

        self.fwd_block = STGCNBlock(in_features=input_dim, hidden_dim=hidden_dim, adj=adj, kernel_size=kernel_size)

        self.bwd_block = STGCNBlock(in_features=input_dim, hidden_dim=hidden_dim, adj=adj, kernel_size=kernel_size)

        self.fuse = nn.Linear(2 * hidden_dim, target_dim)

        self.output_activation = nn.Identity()

    def forward(self, x_past, x_future, t_past, t_future):
        #concatante
        x_past_combined = torch.cat([x_past, t_past], dim=-1)
        x_future_combined = torch.cat([x_future, t_future], dim=-1)

        #forward branch
        past_out = self.fwd_block(x_past_combined)
        state_t_fwd = past_out[:, -1, :, :]  

        #backward branch
        x_rev_combined = x_future_combined.flip(dims=[1])
        future_out = self.bwd_block(x_rev_combined)
        state_t_bwd = future_out[:, -1, :, :] 

        #combine
        combined = torch.cat([state_t_fwd, state_t_bwd], dim=-1)
        out_center = self.fuse(combined)

        return self.output_activation(out_center)

class GraphConv(nn.Module):
    def __init__(self, in_features, out_features, adj):
        super().__init__()
        self.register_buffer("adj", adj)
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        x = torch.matmul(x, self.weight)
        x = torch.einsum("nm,bwmh->bwnh", self.adj, x) 
        return x

class TemporalLayer(nn.Module):
    """
    as defined in paper (stgcn)
    """
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=2 * out_channels, 
            kernel_size=(1, kernel_size),
            padding=(0, kernel_size // 2) 
        )

    def forward(self, x):
        x = self.conv(x)
        P, Q = torch.chunk(x, 2, dim=1)
        return P * torch.sigmoid(Q)

class STGCNBlock(nn.Module):
    """
    based in the stgcn paper
    """
    def __init__(self, in_features, hidden_dim, adj, kernel_size=3):
        super().__init__()

        self.temporal1 = TemporalLayer(
            in_channels=in_features, 
            out_channels=hidden_dim, 
            kernel_size=kernel_size
        )

        self.spatial = GraphConv(
            in_features=hidden_dim,
            out_features=hidden_dim,
            adj=adj
        )
        self.relu = nn.ReLU()

        self.temporal2 = TemporalLayer(
            in_channels=hidden_dim, 
            out_channels=hidden_dim, 
            kernel_size=kernel_size
        )

        if in_features != hidden_dim:
            self.residual = nn.Conv2d(in_features, hidden_dim, kernel_size=1)
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        """
        input x: (B, W, N, F)
        """

        x_in = x.permute(0, 3, 2, 1) # (B, F, N, W)
        
        h = self.temporal1(x_in) # (B, H, N, W)

        h = h.permute(0, 3, 2, 1) # (B, W, N, H)

        h = self.spatial(h)
        h = self.relu(h) 
        h = h.permute(0, 3, 2, 1) # (B, H, N, W)

        h = self.temporal2(h) # (B, H, N, W)

        res = self.residual(x_in)
        
        return (h + res).permute(0, 3, 2, 1) #(B, W, N, H)

