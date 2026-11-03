import torch
import torch.nn.functional as F

def cross_entropy_derivative(logits: torch.Tensor, target: int) -> torch.Tensor:
    """
    Compute the derivative of cross-entropy loss with respect to logits.
    
    Args:
        logits: Raw model outputs tensor
        target: Index of the true class
        
    Returns:
        Gradient tensor
    """
    x = logits.clone().requires_grad_(True)
    loss = F.cross_entropy(x.unsqueeze(0),
                            torch.tensor([target]))
    loss.backward()
    return x.grad

    logits = logits.clone().requires_grad_(True)
    loss = F.cross_entropy(logits.unsqueeze(0),
                            torch.tensor([target]))
    loss.backward()

    return logits.grad