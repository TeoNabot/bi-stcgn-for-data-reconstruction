import torch

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")  # This enables Apple M1/M2/M3 acceleration
    else:
        return torch.device("cpu")