import torch
import torch.nn.functional as F

def train_step(batch, model, optimizer, device):
    """
    Performs a single training step for the BiSTGCN model.
    
    Args:
        batch: Dictionary returned by STBGNNDataset
        model: The BiSTGCN model instance
        optimizer: The optimizer instance
        device: 'cuda' or 'cpu'
        
    Returns:
        loss_val: float, the training loss for this step
    """
    model.train()
    
    # 1. Unpack and Move to Device
    # x_past_masked: (B, W+1, N, F) - Includes time t at the end (masked)
    x_past = batch["x_past_masked"].to(device)
    
    # x_future: (B, W, N, F) - From t+1 to t+W
    x_future = batch["x_future"].to(device)
    
    # target: (B, N, F) - Ground truth at time t
    target = batch["target"].to(device)
    
    # loss_mask: (B, N) - 1.0 for artificially masked nodes, 0.0 otherwise
    loss_mask = batch["loss_mask"].to(device)

    # 2. Forward Pass
    optimizer.zero_grad()
    
    # Model returns reconstruction of time t: (B, N, F)
    reconstruction = model(x_past, x_future)

    # 3. Loss Calculation
    # We use reduction='none' to get the element-wise error first
    # shape: (B, N, F)
    raw_loss = F.mse_loss(reconstruction, target, reduction='none')

    # Reduce over the feature dimension (F) to get error per node
    # shape: (B, N)
    per_node_loss = raw_loss.mean(dim=-1)

    # Apply the mask: Zero out loss for nodes we didn't artificially mask
    # shape: (B, N)
    masked_loss = per_node_loss * loss_mask

    # Normalize loss: Sum of errors / Number of masked nodes
    # We add a small epsilon to avoid division by zero if a batch has no masks
    num_masked_nodes = loss_mask.sum()
    final_loss = masked_loss.sum() / (num_masked_nodes + 1e-6)

    # 4. Backward Pass
    final_loss.backward()
    
    # Gradient clipping (optional but recommended for GNNs/RNNs)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    
    optimizer.step()

    return final_loss.item()