import torch
from typing import Callable, Tuple

def numerical_gradient_check(f: Callable, x: torch.Tensor, analytical_grad: torch.Tensor, epsilon: float = 1e-7) -> Tuple[torch.Tensor, float]:
    """
    Perform numerical gradient checking using centered finite differences.

    Args:
        f: A function that takes a torch.Tensor and returns a scalar
        x: torch.Tensor, the point at which to check gradient
        analytical_grad: torch.Tensor, the analytically computed gradient
        epsilon: float, small value for finite difference approximation

    Returns:
        tuple: (numerical_grad, relative_error)
    """
    x = torch.tensor(x, dtype=torch.float64)
    analytical_grad = torch.tensor(analytical_grad, dtype=torch.float64)
    
    x_flat = x.flatten().clone()
    numerical_grad_flat = torch.zeros(x_flat.shape, dtype=torch.float64)
    for i in range(x_flat.numel()):
        x_p = x_flat.clone()
        x_m = x_flat.clone()

        x_p[i] = x_p[i] + epsilon
        x_m[i] = x_m[i] - epsilon

        f_p = f(x_p.reshape(x.shape))
        f_m = f(x_m.reshape(x.shape))

        numerical_grad_flat[i] = (f_p - f_m) / (2*epsilon)
    
    numerical_grad = numerical_grad_flat.reshape(x.shape)

    diff_norm = torch.norm(numerical_grad - analytical_grad)
    sum_norm = torch.norm(numerical_grad) + torch.norm(analytical_grad)

    relative_error = diff_norm / (sum_norm + epsilon)

    return numerical_grad, relative_error.item()


       
