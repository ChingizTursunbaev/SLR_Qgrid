import torch
import torch.nn as nn

class FeedForwardNN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, **kwargs):
        super().__init__()
        hidden_features = hidden_features or in_features * 4
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, **kwargs)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features, **kwargs)

    def forward(self, x: torch.Tensor):
        return self.fc2(self.act(self.fc1(x)))