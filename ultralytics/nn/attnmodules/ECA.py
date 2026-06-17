import torch
import torch.nn as nn
import math

class ECA(nn.Module):
    def __init__(self, channels:int, k_size:int=None, gamma:int=2, b:int=1):
        super(ECA, self).__init__()
        assert channels > 0
        self.gap = nn.AdaptiveAvgPool2d(1)
        if k_size is None:
            t = int(abs(math.log2(channels) / gamma + b))
            k_size = t if t % 2 else t+1
            k_size = max(3, k_size)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.gap(x)                             # (B, C, 1, 1)
        y = y.squeeze(-1).squeeze(-1)               # (B, C)
        y = y.unsqueeze(1)                          # (B, 1, C)
        y = self.conv1d(y)                          # (B, 1, C)
        y = self.sigmoid(y).squeeze(1).unsqueeze(-1).unsqueeze(-1) # (B, C, 1, 1)
        return x * y

# if __name__ == '__main__':
#     x = torch.randn(25, 1024, 13, 13)
#     eca = ECA(channels=3, k_size=None, gamma=2, b=1)
#     print(eca(x).shape)