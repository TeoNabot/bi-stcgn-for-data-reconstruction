import torch
import torch.nn as nn
import torch.nn.functional as F
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

input tensors have shape (B, N, F, W), where 
B: batch size
N: number of nodes
F: number of features
W+1: window size (W elements at each side of the center time)
"""

class GraphConv(nn.Module):
    def __init__(self, in_features, out_features, adj):
        super().__init__()
        self.register_buffer("adj", adj)
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        """
        x input: (B, N, F, T)
        """
        # 1. Linear Projection (applied to every node/time independently)
        # (B, N, F, T) -> (B, N, OutF, T)
        # Using matmul is fine, but einsum is often cleaner/safer for this
        x_mul = torch.einsum('bnft, fo -> bnot', x, self.weight)

        B, N, F_out, T = x_mul.shape

        # 2. Graph Propagation
        # We need to multiply adj (N, N) with x (..., N, ...)
        # We can merge B and T into a single "batch" dimension to broadcast adj
        
        # Current shape: (B, N, F_out, T)
        # We need N to be the distinct dimension for matmul, and (B, T) to be the batch
        
        # Permute to put B and T together: (B, T, N, F_out)
        x_tmp = x_mul.permute(0, 3, 1, 2) 
        
        # Now we can safely collapse B and T: (B*T, N, F_out)
        x_flat = x_tmp.reshape(B * T, N, F_out)
        
        # (N, N) @ (B*T, N, F_out) -> (B*T, N, F_out)
        out = torch.matmul(self.adj, x_flat)
        
        # 3. Restore Shape
        # Unpack: (B, T, N, F_out)
        out = out.reshape(B, T, N, F_out)
        
        # Permute back to your standard format: (B, N, F_out, T)
        return out.permute(0, 2, 3, 1)

class TemporalConvLayer(nn.Module):
    """
    Temporal Convolution with Gated Linear Unit (GLU)
    Input: (B, N, C, T)
    """
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels * 2, # *2 for GLU (Output = Value * Sigmoid(Gate))
            kernel_size=(1, kernel_size), #height=1 (nodes), width=kernel_size (time). slides for each sensor through time
            padding=(0, kernel_size // 2) #same padding, keep size T
        )

    def forward(self, x):
        # x shape: (B, N, C, T) -> interpreted as (Batch, Channel, Height=N, Width=T) for Conv2d
        #i think of a grid with as sensors and columns as time steps. see that adjancency is not used here.
        # We treat N as the 'Height' dimension (1) and T as 'Width' (kernel moves along T, cause it is defines with size (1, kernel_size))
        x = x.permute(0, 2, 1, 3) # (B, C, N, T)
        x = self.conv(x)          # (B, 2*OutC, N, T)
        x = x.permute(0, 2, 1, 3) # (B, N, 2*OutC, T)
        
        # GLU: split channels
        lhs, rhs = x.chunk(2, dim=2)
        return lhs * torch.sigmoid(rhs) # (B, N, OutC, T)
        
class STGCNBlock(nn.Module):
    def __init__(self, in_channels, hidden_dim, adj, kernel_size=3):
        super().__init__()
        
        # temp conv
        self.tconv1 = TemporalConvLayer(in_channels, hidden_dim, kernel_size)
        
        # spatial graph conv
        self.gconv = GraphConv(hidden_dim, hidden_dim, adj)
        self.relu = nn.ReLU()
        
        # temp conv again
        self.tconv2 = TemporalConvLayer(hidden_dim, hidden_dim, kernel_size)
        
        # residual connection inside block; conv to match dimensions if needed
        self.residual_conv = None
        if in_channels != hidden_dim:
            self.residual_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=(1, 1))

        self.ln = nn.LayerNorm([hidden_dim])

    def forward(self, x):
        """
        x: (B, N, F, T)
        """
        residual = x
        
        # Block structure: Temporal -> Spatial -> Temporal
        x = self.tconv1(x) # (B, N, H, T)
        x = self.relu(self.gconv(x)) # (B, N, H, T)
        x = self.tconv2(x) # (B, N, H, T)

        # Residual connection
        if self.residual_conv is not None:
            # (B, N, F, T) -> permute for Conv2d -> (B, F, N, T)
            res_tmp = residual.permute(0, 2, 1, 3)
            residual = self.residual_conv(res_tmp).permute(0, 2, 1, 3)
            
        x = x + residual
        
        # Layer Norm (applied over the channel dimension last)
        # x is (B, N, F, T), LayerNorm expects [F] or [N, F, T]... 
        # usually we norm over features.
        x = x.permute(0, 1, 3, 2) # (B, N, T, F)
        x = self.ln(x)
        x = x.permute(0, 1, 3, 2) # (B, N, F, T)
        
        return x

class BiSTGCN(nn.Module):
    def __init__(self, in_features, hidden_dim, adj, kernel_size=3):
        super().__init__()
        
        # Ensure adj is a buffer (not a trainable parameter, but part of state_dict)
        if not isinstance(adj, torch.Tensor):
             adj = torch.tensor(adj)
        self.register_buffer('adj', adj)
        
        # Branch 1: Past -> Current
        self.fwd_block = STGCNBlock(in_features, hidden_dim, self.adj, kernel_size)
        
        # Branch 2: Future -> Current
        self.bwd_block = STGCNBlock(in_features, hidden_dim, self.adj, kernel_size)
        
        # Fusion: Combines Past representation and Future representation
        self.fuse = nn.Linear(2 * hidden_dim, in_features)

    def forward(self, x_past, x_future):
        """
        Input Requirements:
        -------------------
        x_past:   (B, W+1, N, F) -> Sequence [t-W ... t]
                  * Must include time t *
                  * Missing nodes at t should be masked. In training, mask also target nodes at timestep t *
                  
        x_future: (B, W+1, N, F)  -> Sequence [t ... t+W]
                  * starts at time t *
                  * Missing nodes at t should be masked (same as x_past). In training, mask also target nodes at timestep t *
        """

        # 1. Prepare Data
        # Permute to (B, N, F, T) for our Conv2d-based blocks
        # x_past becomes: [t-W, ..., t]
        x_p = x_past.permute(0, 2, 3, 1)   
        
        # x_future becomes: [t, ..., t+W]
        x_f = x_future.permute(0, 2, 3, 1) 

        # ---------------------------------------------------------
        # 2. Forward Branch (Past context + Spatial info at t)
        # ---------------------------------------------------------
        # Processing sequence: [t-W, ..., t]
        # The GraphConv inside this block mixes neighbors at every step, including t.
        past_out = self.fwd_block(x_p) # Output: (B, N, H, T)
        
        # We extract the LAST element, which represents the processed state at time t
        state_t_past = past_out[..., -1] # (B, N, H)


        # ---------------------------------------------------------
        # 3. Backward Branch (Future context + Spatial info at t)
        # ---------------------------------------------------------
        # We want to process from t+W down to t.
        # Current x_f is [t, t+1, ..., t+W].
        # We flip it to: [t+W, ..., t+1, t].
        x_f_flipped = x_f.flip(dims=[-1])
        
        # Processing sequence: [t+W, ..., t]
        future_out = self.bwd_block(x_f_flipped)
        
        # We extract the LAST element. 
        # Since we flipped the input, the last element corresponds to time t.
        state_t_future = future_out[..., -1] # (B, N, H)


        # ---------------------------------------------------------
        # 4. Fusion and Reconstruction
        # ---------------------------------------------------------
        # Now we have two views of time t:
        # 1. state_t_past: What t looks like based on history + current neighbors
        # 2. state_t_future: What t looks like based on future + current neighbors
        
        combined = torch.cat([state_t_past, state_t_future], dim=-1) # (B, N, 2H)
        
        # Project back to feature space
        reconstruction = self.fuse(combined) # (B, N, F)
        
        return reconstruction