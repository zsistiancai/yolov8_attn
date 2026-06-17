import torch
import torch.nn as nn

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        assert channels > 0
        hidden = channels // reduction

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )

        pad = kernel_size // 2
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=pad, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.avg_pool(x)
        mx = self.max_pool(x)

        Mc = self.sigmoid(self.mlp(avg) + self.mlp(mx))
        x_c = x * Mc

        avg_c = x_c.mean(dim=1, keepdim=True)
        max_c = x_c.max(dim=1, keepdim=True)[0]
        s = torch.cat([avg_c, max_c], dim=1)
        Ms = self.sigmoid(self.spatial_conv(s))
        out = x_c * Ms

        return out

# if __name__ == '__main__':
#     x = torch.randn(10, 256, 200, 200)
#     model = CBAM(channels=256, reduction=16, kernel_size=7)
#     print(model(x).shape)
