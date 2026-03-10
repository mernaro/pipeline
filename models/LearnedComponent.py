import torch
import torch.nn as nn
import math


def truncated_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    with torch.no_grad():
        size = tensor.shape
        tmp = tensor.new_empty(size + (4,)).normal_()
        valid = (tmp > a) & (tmp < b)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.data.mul_(std).add_(mean)
    return tensor

def conv_init(m, std):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        truncated_normal_(m.weight, mean=0.0, std=std)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.001)
    elif isinstance(m, nn.Linear):
        truncated_normal_(m.weight, mean=0.0, std=0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=pad)
        self.elu = nn.ELU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=pad)
        

    def forward(self, x):
        out = self.conv1(x)
        out = self.elu(out)
        out = self.conv2(out)
        return out + x

class LearnedComponent(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=64, kernel_size=3,init_variance=0.05):
        super().__init__()
        self.in_ch = in_channels
        self.base_ch = hidden_channels
        self.kernel_size = kernel_size
        self.init_var = init_variance

        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size, padding=1),
            nn.ELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=1),
            nn.ELU()
        )

        # store residual blocks
        self.resblocks = nn.ModuleList([ResidualBlock(hidden_channels) for _ in range(10)])

        self.dec = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=3, padding=1)
        )

        self.apply(lambda m: conv_init(m, std=math.sqrt(self.init_var)))

    def network(self, input, is_training=True, n_residual_blocks: int = 2):
        out = self.enc(input)
        for i in range(n_residual_blocks):
            out = self.resblocks[i](out)
        out = self.dec(out)
        return out


