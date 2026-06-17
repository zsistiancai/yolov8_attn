import torch.nn as nn
import torch

class SimAM(nn.Module):
    def __init__(self, channels:int, e_lambda:float=1e-4):
        super(SimAM, self).__init__()
        # self.channels = channels
        self.e_lambda = e_lambda
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        mu = x.mean(dim=(2, 3), keepdim=True)
        d = (x - mu).pow(2)

        v = d.mean(dim=(2, 3), keepdim=True)
        score = d / (4.0 * (v + self.e_lambda)) + 0.5
        attn = self.sigmoid(score)

        return x * attn

# if __name__ == '__main__':
#     x = torch.randn(10, 512, 512, 512)
#     model = SimAM(512)
#     print(model(x).shape)