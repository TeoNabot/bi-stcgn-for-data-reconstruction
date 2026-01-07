import torch
import torch.nn.functional as F

def train_step(batch, model, optimizer, device):
    """
    Performs a single training step for the BiSTGCN model.
    """
    model.train()
    
    # 1. Unpack and Move to Device
    # ------------------------------------------------------------------
    # NOTE: Keys must match the return dictionary of STBGNNDataset
    # Shapes coming from DataLoader will be (Batch, Time, Nodes, Features)
    # ------------------------------------------------------------------
    
    # x_past: Includes time t at the end (masked)
    x_past = batch["x_past_masked"].to(device)  # Shape: (B, W_past, N, F)
    
    # x_future: Includes time t at the start (masked)
    x_future = batch["x_future_masked"].to(device) # Shape: (B, W_future, N, F)
    
    # target: Ground truth at time t
    target = batch["target"].to(device)     # Shape: (B, N, F)
    
    # loss_mask: 1.0 for artificially masked nodes, 0.0 otherwise
    loss_mask = batch["loss_mask"].to(device) # Shape: (B, N)

    # 2. Forward Pass
    # ------------------------------------------------------------------
    optimizer.zero_grad()
    
    # Model inputs: (B, T, N, F)
    # Model reconstructs time t using both directions
    reconstruction = model(x_past, x_future) # Output: (B, N, F)

    # 3. Loss Calculation
    # ------------------------------------------------------------------
    # Calculate squared error for every element: (B, N, F)
    # We use reduction='none' to handle the mask manually
    mse_loss = F.mse_loss(reconstruction, target, reduction='none')

    # average over features; note that input is normalized
    node_error = mse_loss.mean(dim=-1)

    # Apply the mask: Zero out errors for nodes that were NOT masked
    # We only want to learn from the "holes" we created
    masked_error = node_error * loss_mask # (B, N)

    # Normalize: Sum of Error / Number of Masked Nodes
    # Adding epsilon to denominator to prevent NaN if a batch has 0 masks
    num_masked = loss_mask.sum()
    final_loss = masked_error.sum() / (num_masked + 1e-6)

    # 4. Backward Pass
    # ------------------------------------------------------------------
    final_loss.backward()
    
    # Clip gradients to prevent exploding gradients in deep GNNs/RNNs
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    
    optimizer.step()

    return final_loss.item()