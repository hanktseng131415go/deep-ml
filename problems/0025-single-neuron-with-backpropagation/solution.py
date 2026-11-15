import torch
import torch.nn as nn
import torch.nn.functional as F

def train_neuron(features: torch.Tensor, labels: torch.Tensor, initial_weights: torch.Tensor, initial_bias: float, learning_rate: float, epochs: int) -> tuple[list[float], float, list[float]]:
    features = torch.tensor(features)
    labels = torch.tensor(labels)
    weights = torch.tensor(initial_weights)

    net = nn.Linear(features.shape[1], 1, bias=True)
    with torch.no_grad():
        net.weight.copy_(weights.unsqueeze(0))
        net.bias.fill_(initial_bias)

    opt = torch.optim.SGD(net.parameters(), lr=learning_rate)

    mse_values = []
    
    for _ in range(epochs):

        y = net(features)
        y = torch.sigmoid(y).squeeze()
        loss = F.mse_loss(y, labels)
        mse_values.append(round(loss.item(), 4))
        loss.backward()
        opt.step()
        opt.zero_grad()

    updated_weights = [round(w, 4) for w in net.weight.data.squeeze().tolist()]
    updated_bias = round(net.bias.item(), 4)

    return updated_weights, updated_bias, mse_values