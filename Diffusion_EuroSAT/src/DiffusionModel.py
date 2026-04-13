import torch

def normalize_eurosat(x, mean, std):
    x = x.float() / 255.0
    mean = torch.tensor(mean).view(1, 3, 1, 1)
    std = torch.tensor(std).view(1, 3, 1, 1)
    return (x - mean) / std

def denormalize_eurosat(x, mean, std):
    mean = torch.tensor(mean).view(1, 3, 1, 1)
    std = torch.tensor(std).view(1, 3, 1, 1)
    x = x * std + mean
    return torch.clamp(x * 255.0, 0, 255).byte()

def forward_diffusion(x_0, t):
    """
    Implements the forward diffusion process for a given input image x_0 and time step t.
    """
    t = t.view(-1, 1, 1, 1)
    mean_coef = torch.exp(-t)
    var_coef = 1 - torch.exp(-2 * t)
    std = torch.sqrt(var_coef)
    
    epsilon = torch.randn_like(x_0)
    x_t = mean_coef * x_0 + std * epsilon
    
    return x_t, epsilon