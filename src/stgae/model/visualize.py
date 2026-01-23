import matplotlib.pyplot as plt
import torch
import numpy as np

def visualize_sensor_reconstruction(dataloader, model, scaler, sensor_idx, feature_names, device, num_steps=2000):
    """
    Plots a direct comparison of Ground Truth vs. Prediction ONLY for masked nodes.
    Optimized for dense datasets (thousands of points).
    """
    model.eval()
    
    collected_targets_real = []
    collected_preds_real = []
    collected_masks = []
    
    steps_collected = 0

    with torch.no_grad():
        for batch in dataloader:
            if steps_collected >= num_steps:
                break

            x_past = batch["x_past_masked"].to(device)
            x_future = batch["x_future_masked"].to(device)
            target = batch["target"].to(device)
            loss_mask = batch["loss_mask"].to(device)
            
            t_past = batch["time_past"].to(device)
            t_future = batch["time_future"].to(device)

            #expand Time Features
            B, W, N, _ = x_past.shape
            t_past_expanded = t_past.unsqueeze(2).expand(-1, -1, N, -1)
            t_future_expanded = t_future.unsqueeze(2).expand(-1, -1, N, -1)

            #predict
            recon_norm = model(x_past, x_future, t_past_expanded, t_future_expanded)

            #inverse transform
            recon_real = scaler.inverse_transform(recon_norm)
            target_real = scaler.inverse_transform(target)

            # extract data for specific sensor
            collected_preds_real.append(recon_real[:, sensor_idx, :].cpu().numpy())
            collected_targets_real.append(target_real[:, sensor_idx, :].cpu().numpy())
            collected_masks.append(loss_mask[:, sensor_idx].cpu().numpy())
            
            steps_collected += B

    full_preds = np.concatenate(collected_preds_real, axis=0)[:num_steps]
    full_targets = np.concatenate(collected_targets_real, axis=0)[:num_steps]
    full_masks = np.concatenate(collected_masks, axis=0)[:num_steps]

    #plot
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    time_axis = np.arange(len(full_targets))

    for f_idx, ax in enumerate(axes):
        feature_name = feature_names[f_idx]
        
        masked_indices = np.where(full_masks == 1)[0]
        
        if len(masked_indices) > 0:
            gt_values = full_targets[masked_indices, f_idx]
            pred_values = full_preds[masked_indices, f_idx]
            time_points = time_axis[masked_indices]

            #error bars
            ax.vlines(
                x=time_points, 
                ymin=np.minimum(gt_values, pred_values), 
                ymax=np.maximum(gt_values, pred_values), 
                color='gray', 
                alpha=0.3,      
                linewidth=0.5,  
                label='_nolegend_'
            )

            #gt
            ax.scatter(
                time_points, 
                gt_values, 
                color='black', 
                label="Ground Truth", 
                alpha=0.6, 
                s=10,            
                marker='.'     
            )

            #predictin
            ax.scatter(
                time_points, 
                pred_values, 
                color='red', 
                label="Prediction", 
                alpha=0.6, 
                s=20,            
                marker='x',
                linewidths=0.8   
            )

        ax.set_title(f"{feature_name} (Masked Points Only)", fontsize=10, fontweight='bold')
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.2)
        
        if f_idx == 0:
            lgnd = ax.legend(loc="upper right")
            #increase marker size in legend for readability
            for handle in lgnd.legend_handles:
                handle.set_sizes([30.0])

    plt.xlabel("Time Steps")
    plt.tight_layout()
    plt.show()

