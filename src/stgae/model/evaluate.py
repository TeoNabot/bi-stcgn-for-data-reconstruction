import torch
import torch.nn.functional as F
import numpy as np

def evaluate(dataloader, model, device):
    """
    Evaluates the model on the validation set.
    Returns: Average Loss (MSE), MAE, and RMSE over all batches.
    """
    model.eval() # Set model to evaluation mode
    
    total_loss = 0.0
    total_mae = 0.0
    total_rmse = 0.0
    batch_count = 0

    with torch.no_grad(): # Disable gradient calculation
        for batch in dataloader:
            # 1. Unpack (Same as train_step)
            x_past = batch["x_past_masked"].to(device)
            x_future = batch["x_future_masked"].to(device)
            target = batch["target"].to(device)
            loss_mask = batch["loss_mask"].to(device) # 1.0 for missing nodes

            # 2. Forward Pass
            reconstruction = model(x_past, x_future)

            # 3. Calculate Metrics only on Masked Nodes (The "Holes")
            # We create a boolean mask to select only the relevant indices
            # loss_mask is (B, N), target/recon is (B, N, F)
            # We expand mask to match feature dim for easy selection
            mask_bool = loss_mask.unsqueeze(-1).expand_as(target).bool()
            
            # Flatten and select only the masked values
            pred_masked = reconstruction[mask_bool]
            true_masked = target[mask_bool]

            if len(true_masked) > 0:
                # MSE (Standard Loss)
                mse = F.mse_loss(pred_masked, true_masked).item()
                
                # MAE (Mean Absolute Error)
                mae = F.l1_loss(pred_masked, true_masked).item()
                
                # RMSE (Root Mean Squared Error)
                rmse = torch.sqrt(F.mse_loss(pred_masked, true_masked)).item()

                total_loss += mse
                total_mae += mae
                total_rmse += rmse
                batch_count += 1

    # Avoid division by zero if dataloader is empty
    if batch_count == 0:
        return 0.0, 0.0, 0.0

    return total_loss / batch_count, total_mae / batch_count, total_rmse / batch_count