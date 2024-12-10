import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 64, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 10, bias=False)

    def forward(self, x):
        if not self.training:
            assert torch.all(x > -1.1) and torch.all(x < 1.1)
        x = x.view(-1, 28 * 28)
        if not self.training:
            assert torch.all(x > -1.1) and torch.all(x < 1.1)
        x = self.fc1(x)
        if not self.training:
            assert torch.all(x > -1.1) and torch.all(x < 1.1)
        x = self.relu1(x)
        if not self.training:
            assert torch.all(x > -1.1) and torch.all(x < 1.1)
        x = self.fc2(x)
        return x
