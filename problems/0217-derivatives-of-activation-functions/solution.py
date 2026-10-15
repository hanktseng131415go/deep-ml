import torch
import torch.nn.functional as F

def activation_derivatives(x: float) -> dict[str, float]:
    """
    Compute the derivatives of Sigmoid, Tanh, and ReLU at a given point x
    using PyTorch autograd.
    
    Args:
        x: Input value
        
    Returns:
        Dictionary with keys 'sigmoid', 'tanh', 'relu' and their derivative values
    """
    # Your code here - use autograd!
    # Hint: Create tensors with requires_grad=True, apply activation, call .backward()
    x_sigmoid = torch.tensor(x, requires_grad=True)
    y_sigmoid = F.sigmoid(x_sigmoid)
    y_sigmoid.backward()
    dy_sigmoid = x_sigmoid.grad.item()

    x_tanh = torch.tensor(x, requires_grad=True)
    y_tanh = F.tanh(x_tanh)
    y_tanh.backward()
    dy_tanh = x_tanh.grad.item()

    x_relu = torch.tensor(x, requires_grad=True)
    y_relu = F.relu(x_relu)
    y_relu.backward()
    dy_relu = x_relu.grad.item()

    return {
        'sigmoid': dy_sigmoid,
        'tanh': dy_tanh,
        'relu': dy_relu
    }

