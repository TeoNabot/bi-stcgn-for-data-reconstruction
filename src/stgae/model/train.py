
import torch
import torch.nn.functional as F

def train_step(batch, model, optimizer, device):
    model.train()
    
    x_past = batch["x_past_masked"].to(device)     #(B, W+1, N, F_target)
    x_future = batch["x_future_masked"].to(device) 
    
    target = batch["target"].to(device)            #(B, N, F_target)
    loss_mask = batch["loss_mask"].to(device)      
    
    t_past = batch["time_past"].to(device)        
    t_future = batch["time_future"].to(device)    
    
    B, W, N, _ = x_past.shape
    
    #unsqueeze dim 2 to get (B, W+1, 1, F_time), expand dim to N to get (B, W+1, N, F_time)
    t_past_expanded = t_past.unsqueeze(2).expand(-1, -1, N, -1)
    t_future_expanded = t_future.unsqueeze(2).expand(-1, -1, N, -1)

    #forward pass
    optimizer.zero_grad()

    reconstruction = model(x_past, x_future, t_past_expanded, t_future_expanded) 

    #loss
    mse_loss = F.mse_loss(reconstruction, target, reduction='none') 

    node_error = mse_loss.mean(dim=-1) 

    masked_error = node_error * loss_mask 
    num_masked = loss_mask.sum()
    final_loss = masked_error.sum() / (num_masked + 1e-6)

    #backward àss
    final_loss.backward()
    
    #clip gradients to ensure stability
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    
    optimizer.step()

    return final_loss.item()



