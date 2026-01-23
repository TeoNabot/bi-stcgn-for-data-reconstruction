import torch
import torch.nn.functional as F
import numpy as np

def evaluate(dataloader, model, device):
    """
    evaluates the model on the validation set using masked nodes.
    returns: Average Loss (MSE), MAE, and RMSE over all batches.
    """
    model.eval()
    
    total_loss = 0.0
    total_mae = 0.0
    total_rmse = 0.0
    batch_count = 0

    with torch.no_grad():
        for batch in dataloader:
            x_past = batch["x_past_masked"].to(device)    # (B, W+1, N, F_target)
            x_future = batch["x_future_masked"].to(device) # (B, W+1, N, F_target)
            
            target = batch["target"].to(device)            # (B, N, F_target)
            loss_mask = batch["loss_mask"].to(device)      # (B, N) -> 1.0 where masked

            t_past = batch["time_past"].to(device)         # (B, W+1, F_time)
            t_future = batch["time_future"].to(device)     # (B, W+1, F_time)

            B, W, N, _ = x_past.shape
            
            t_past_expanded = t_past.unsqueeze(2).expand(-1, -1, N, -1)
            t_future_expanded = t_future.unsqueeze(2).expand(-1, -1, N, -1)

            #forward pass
            reconstruction = model(x_past, x_future, t_past_expanded, t_future_expanded)

            mask_bool = loss_mask.unsqueeze(-1).expand_as(target).bool()
            
            # flattens the tensors to arrays of size (masked * F)
            pred_masked = reconstruction[mask_bool]
            true_masked = target[mask_bool]

            if len(true_masked) > 0:
                mse = F.mse_loss(pred_masked, true_masked).item()

                mae = F.l1_loss(pred_masked, true_masked).item()
                rmse = torch.sqrt(F.mse_loss(pred_masked, true_masked)).item()

                total_loss += mse
                total_mae += mae
                total_rmse += rmse
                batch_count += 1

    if batch_count == 0:
        return 0.0, 0.0, 0.0

    return total_loss / batch_count, total_mae / batch_count, total_rmse / batch_count


# def predict(dataloader, model, scaler, device):
#     """
#     Runs inference on the test set, rescales predictions to original units,
#     and computes metrics.
    
#     Args:
#         dataloader: Test dataloader
#         model: Trained BiSTGCN model
#         scaler: Your custom StandardScaler object (must match training scaler)
#         device: torch.device
        
#     Returns:
#         metrics: Dict containing 'mse', 'mae', 'rmse' in original units
#         outputs: Dict containing 'preds' and 'targets' (flattened, rescaled arrays)
#     """
#     model.eval()
    
#     # Storage for all batches (to compute global metrics or for plotting later)
#     all_preds = []
#     all_targets = []
    
#     with torch.no_grad():
#         for batch in dataloader:
#             # 1. Unpack Inputs (Same as evaluate)
#             # ------------------------------------------------------------------
#             x_past = batch["x_past_masked"].to(device)
#             x_future = batch["x_future_masked"].to(device)
#             target = batch["target"].to(device)       # Normalized Ground Truth
#             loss_mask = batch["loss_mask"].to(device) # Mask (1.0 = missing)

#             t_past = batch["time_past"].to(device)
#             t_future = batch["time_future"].to(device)

#             # 2. Expand Time Features (Broadcasting)
#             # ------------------------------------------------------------------
#             B, W, N, _ = x_past.shape
#             t_past_expanded = t_past.unsqueeze(2).expand(-1, -1, N, -1)
#             t_future_expanded = t_future.unsqueeze(2).expand(-1, -1, N, -1)

#             # 3. Forward Pass (Result is Normalized)
#             # ------------------------------------------------------------------
#             # reconstruction shape: (B, N, F_target)
#             reconstruction_norm = model(x_past, x_future, t_past_expanded, t_future_expanded)

#             # 4. Rescale to Original Units
#             # ------------------------------------------------------------------
#             # We must inverse_transform the ENTIRE tensor first because
#             # the scaler expects (..., F) shape.
            
#             # Inverse transform Preds
#             reconstruction_real = scaler.inverse_transform(reconstruction_norm)
            
#             # Inverse transform Targets
#             target_real = scaler.inverse_transform(target)

#             # 5. Select Only Masked Values
#             # ------------------------------------------------------------------
#             # Expand mask to (B, N, F_target)
#             mask_bool = loss_mask.unsqueeze(-1).expand_as(target).bool()
            
#             # Flatten and filter: We only care about the values we masked out
#             pred_flat = reconstruction_real[mask_bool]
#             target_flat = target_real[mask_bool]

#             # Store in list (move to CPU to save GPU memory)
#             if len(target_flat) > 0:
#                 all_preds.append(pred_flat.cpu())
#                 all_targets.append(target_flat.cpu())

#     # 6. Aggregate and Compute Final Metrics
#     # ------------------------------------------------------------------
#     # Concatenate all batches into one huge 1D array
#     if len(all_targets) == 0:
#         return {}, {}
        
#     final_preds = torch.cat(all_preds)
#     final_targets = torch.cat(all_targets)

#     # Compute metrics in Original Units
#     mse = F.mse_loss(final_preds, final_targets).item()
#     mae = F.l1_loss(final_preds, final_targets).item()
#     rmse = torch.sqrt(F.mse_loss(final_preds, final_targets)).item()

#     print(f"Test Results (Rescaled): MAE: {mae:.4f} | RMSE: {rmse:.4f}")

#     return {
#         "mse": mse,
#         "mae": mae,
#         "rmse": rmse
#     }, {
#         "preds": final_preds.numpy(),
#         "targets": final_targets.numpy()
#     }

# import torch
# import torch.nn.functional as F
# import numpy as np

# def predict_per_feature(dataloader, model, scaler, device, feature_names=None):
#     """
#     Runs inference and returns metrics/predictions separate for each feature
#     (e.g., Temp, Humidity) in their original scales.
    
#     Args:
#         feature_names: Optional list of strings, e.g. ["Temp", "Humid", "Volt", "Light"]
#     """
#     model.eval()
    
#     # We will initialize this storage after seeing the first batch to know num_features
#     # Structure: { feature_idx: {'preds': [], 'targets': []} }
#     results_storage = {}
    
#     with torch.no_grad():
#         for batch in dataloader:
#             # 1. Unpack Inputs
#             x_past = batch["x_past_masked"].to(device)
#             x_future = batch["x_future_masked"].to(device)
#             target = batch["target"].to(device)
#             loss_mask = batch["loss_mask"].to(device) # (B, N)

#             t_past = batch["time_past"].to(device)
#             t_future = batch["time_future"].to(device)

#             # 2. Expand Time Features
#             B, W, N, F_target = x_past.shape
#             t_past_expanded = t_past.unsqueeze(2).expand(-1, -1, N, -1)
#             t_future_expanded = t_future.unsqueeze(2).expand(-1, -1, N, -1)

#             # 3. Forward Pass (Normalized Output)
#             # Shape: (B, N, F_target)
#             reconstruction_norm = model(x_past, x_future, t_past_expanded, t_future_expanded)

#             # 4. Rescale to Original Units
#             # INVERSE TRANSFORM the whole tensor first
#             reconstruction_real = scaler.inverse_transform(reconstruction_norm)
#             target_real = scaler.inverse_transform(target)

#             # 5. Loop per Feature to filter masked values separately
#             # If results dict is empty, initialize it
#             if not results_storage:
#                 for f in range(F_target):
#                     results_storage[f] = {'preds': [], 'targets': []}

#             # We use the mask (B, N) which applies to all features for that node
#             mask_bool = loss_mask.bool() # (B, N)

#             for f in range(F_target):
#                 # Slice the specific feature: (B, N)
#                 pred_f = reconstruction_real[:, :, f]
#                 true_f = target_real[:, :, f]

#                 # Select only masked nodes
#                 p_valid = pred_f[mask_bool]
#                 t_valid = true_f[mask_bool]

#                 if len(t_valid) > 0:
#                     results_storage[f]['preds'].append(p_valid.cpu())
#                     results_storage[f]['targets'].append(t_valid.cpu())

#     # 6. Aggregate and Compute Metrics per Feature
#     final_results = {}
    
#     print("-" * 60)
#     print(f"{'Feature':<15} | {'MAE':<10} | {'RMSE':<10} | {'Scale Unit'}")
#     print("-" * 60)

#     for f in results_storage:
#         # Concatenate all batches for this feature
#         if len(results_storage[f]['targets']) == 0:
#             continue
            
#         all_preds = torch.cat(results_storage[f]['preds'])
#         all_targets = torch.cat(results_storage[f]['targets'])
        
#         # Calculate Metrics
#         mae = F.l1_loss(all_preds, all_targets).item()
#         rmse = torch.sqrt(F.mse_loss(all_preds, all_targets)).item()
        
#         # Determine name
#         fname = feature_names[f] if feature_names else f"Feature {f}"
        
#         print(f"{fname:<15} | {mae:.4f}     | {rmse:.4f}     | (Original)")
        
#         final_results[fname] = {
#             "mae": mae,
#             "rmse": rmse,
#             "preds": all_preds.numpy(),
#             "targets": all_targets.numpy()
#         }
#     print("-" * 60)

#     return final_results