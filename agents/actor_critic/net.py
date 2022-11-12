import torch
import torch.nn as nn


class ActorCriticNNModule(nn.Module):
    def __init__(
        self,
        space_dim,
        n_actions,
        hidden_dim: int = 32,
        n_bins=33,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__()
        n_output = n_actions + 1
        self.linear = nn.Sequential(
            nn.Linear(space_dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_output * n_bins),
        )
        self.softmax = nn.Softmax(dim=1)
        self.n_actions = n_actions
        self.n_output = n_output
        self.n_bins = n_bins
        self.device = device
        self.bin_tensor = torch.arange(1, n_bins + 1).float().view(1, 1, -1).to(device)

    def forward(self, x):
        B = x.size(0)
        x = x.to(self.device)
        x = self.linear(x)
        x = x.view(B, self.n_output, self.n_bins)
        x = x * self.bin_tensor
        x = x.mean(dim=2)
        action = self.softmax(x[:, :self.n_actions])
        value = x[:, -1]
        return action, value

    def sync(self, global_module):
        for p, gp in zip(self.parameters(), global_module.parameters()):
            p.data = gp.data.clone()


if __name__ == "__main__":
    net = ActorCriticNNModule(2, 4)
    action, value = net(torch.rand(64, 2))
    print(action.shape)  # (64, 4)
    print(value.shape) # (64)
