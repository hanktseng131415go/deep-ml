import torch
import torch.nn.functional as F
def GeLU(x: torch.Tensor) -> torch.Tensor:
    # Your code here
    return torch.round(F.gelu(torch.tensor(x), approximate= 'tanh'), decimals=4)