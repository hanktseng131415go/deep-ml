import torch

def calculate_covariance_matrix(vectors) -> torch.Tensor:
    """
    Calculate the covariance matrix for given feature vectors using PyTorch.
    Input: 2D array-like of shape (n_features, n_observations).
    Returns a tensor of shape (n_features, n_features).
    """
    v_t = torch.as_tensor(vectors, dtype=torch.float)
    # Your implementation here
    v_mean = torch.mean(v_t, dim=1, keepdim=True)
    v_diff = v_t - v_mean
    num = v_t.shape[1]
    if num > 1:
        cov_matrix = (v_diff @ v_diff.T) / (num - 1)
    else:
        cov_matrix = torch.zeros((v_t.shape[0], v_t.shape[0]))
    
    return cov_matrix