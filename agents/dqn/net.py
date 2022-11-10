import torch
import torch.nn as nn


class NNModule(nn.Module):
    def __init__(
        self,
        space_dim,
        n_actions,
        hidden_dim: int = 32,
        n_bins=33,
        device: str = "cpu",
        **kwargs
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(space_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions * n_bins),
        )
        self.n_actions = n_actions
        self.n_bins = n_bins
        self.device = device
        self.bin_tensor = torch.arange(1, n_bins + 1).float().view(1, 1, -1).to(device)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        B = x.size(0)
        x = x.to(self.device)
        x = self.net(x)
        x = x.view(B, self.n_actions, self.n_bins)
        x = x * self.bin_tensor
        x = x.mean(dim=2)
        return x
