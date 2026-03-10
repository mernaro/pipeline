import torch
import torch.nn

import numpy

class CircularConv2d(torch.nn.Conv2d):

    def __init__(self, 
            in_channels, 
            out_channels, 
            kernel_size, 
            stride=1, 
            dilation=1, 
            groups=1, 
            bias=True, 
            padding_mode='zeros', 
            device=None, 
            dtype=None
        ) -> None:
        
        self.kernel_size = kernel_size
        
        super(CircularConv2d, self).__init__(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding='valid', 
            dilation=dilation, 
            groups=groups, 
            bias=bias, 
            padding_mode=padding_mode, 
            device=device, 
            dtype=dtype
        )
        
    def T(self, x: torch.Tensor) -> torch.Tensor:
        x_flipped = torch.flip(x, dims=[0, 1])
        out = self.forward(x_flipped)
        return torch.flip(out, dims=[0, 1])
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = self.kernel_size[0] // 2
        m = self.kernel_size[1] // 2
        pad_width = [ (0, 0), (n, n), (m, m) ]
        x_padded = numpy.pad(x.cpu().detach().numpy(), pad_width=pad_width , mode='wrap')
        x_padded = torch.tensor(x_padded, dtype= x.dtype, device=x.device)
        return super().forward(x_padded)