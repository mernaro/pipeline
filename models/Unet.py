import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, dropout_rate: float = 0.1):
        super().__init__()
        p = dropout_rate
        b = 8
        # Encodeur
        self.enc1 = self._enc_block(1,     b,    p)
        self.enc2 = self._enc_block(b,     b*2,  p)
        self.enc3 = self._enc_block(b*2,   b*4,  p)
        self.enc4 = self._enc_block(b*4,   b*8,  p)
        self.pool = nn.MaxPool2d(2)
        # Bottleneck
        self.bottleneck = self._enc_block(b*8, b*16, p)
        # Decodeur
        self.up4  = nn.ConvTranspose2d(b*16, b*16, kernel_size=2, stride=2)
        self.dec4 = self._dec_block(b*16 + b*8,  b*8,  p)
        self.up3  = nn.ConvTranspose2d(b*8,  b*8,  kernel_size=2, stride=2)
        self.dec3 = self._dec_block(b*8  + b*4,  b*4,  p)
        self.up2  = nn.ConvTranspose2d(b*4,  b*4,  kernel_size=2, stride=2)
        self.dec2 = self._dec_block(b*4  + b*2,  b*2,  p)
        self.up1  = nn.ConvTranspose2d(b*2,  b*2,  kernel_size=2, stride=2)
        self.dec1 = self._dec_block(b*2  + b,     b,    p)
        # Tete de segmentation
        self.head = nn.Conv2d(b, 1, kernel_size=1)

    @staticmethod
    def _enc_block(in_ch, out_ch, p):
        return nn.ModuleDict({
            'conv1': nn.Conv2d(in_ch,  out_ch, kernel_size=3, padding=1),
            'bn1':   nn.BatchNorm2d(out_ch),
            'drop':  nn.Dropout2d(p),
            'conv2': nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            'bn2':   nn.BatchNorm2d(out_ch),
        })

    @staticmethod
    def _dec_block(in_ch, out_ch, p):
        return nn.ModuleDict({
            'conv1': nn.Conv2d(in_ch,  out_ch, kernel_size=3, padding=1),
            'bn1':   nn.BatchNorm2d(out_ch),
            'drop':  nn.Dropout2d(p),
            'conv2': nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            'bn2':   nn.BatchNorm2d(out_ch),
        })

    def _enc_forward(self, block, x):
        x    = F.relu(block['bn1'](block['conv1'](x)))
        x    = block['drop'](x)
        x    = F.relu(block['bn2'](block['conv2'](x)))
        skip = x
        x    = self.pool(x)
        return skip, x

    @staticmethod
    def _dec_forward(block, x, skip):
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        x = F.relu(block['bn1'](block['conv1'](x)))
        x = block['drop'](x)
        x = F.relu(block['bn2'](block['conv2'](x)))
        return x

    @staticmethod
    def _to_unet_input(tensor):
        """
        Convertit n importe quelle shape en (1, 1, H, W) pour le UNet.
        Gere les cas : (H,W) / (1,H,W) / (H,W,1) / (1,1,H,W) / (B,C,H,W)
        """
        t = tensor

        # Retire les dimensions batch superflues si deja 4D
        if t.dim() == 4:
            # (B, C, H, W) -> prend le premier element et premier canal
            t = t[0, 0:1, :, :]           # (1, H, W)
            t = t.unsqueeze(0)             # (1, 1, H, W)
            return t

        # (H, W, C) -> (C, H, W)
        if t.dim() == 3 and t.shape[2] < t.shape[0] and t.shape[2] < t.shape[1]:
            t = t.permute(2, 0, 1)

        # (C, H, W) ou (1, H, W)
        if t.dim() == 3:
            t = t[0:1, :, :]              # garde un seul canal (1, H, W)
            t = t.unsqueeze(0)            # (1, 1, H, W)
            return t

        # (H, W)
        if t.dim() == 2:
            t = t.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            return t

        raise ValueError(f"Shape non supportee pour UNet : {tensor.shape}")

    def forward(self, x):
        # Normalisation de l entree -> toujours (1, 1, H, W)
        x = self._to_unet_input(x)
        x = torch.clamp(x, 0.0, 1.0)

        # Encodeur
        skip1, x = self._enc_forward(self.enc1, x)
        skip2, x = self._enc_forward(self.enc2, x)
        skip3, x = self._enc_forward(self.enc3, x)
        skip4, x = self._enc_forward(self.enc4, x)

        # Bottleneck (sans pool sur la sortie)
        x = F.relu(self.bottleneck['bn1'](self.bottleneck['conv1'](x)))
        x = self.bottleneck['drop'](x)
        x = F.relu(self.bottleneck['bn2'](self.bottleneck['conv2'](x)))

        # Decodeur
        x = self._dec_forward(self.dec4, self.up4(x), skip4)
        x = self._dec_forward(self.dec3, self.up3(x), skip3)
        x = self._dec_forward(self.dec2, self.up2(x), skip2)
        x = self._dec_forward(self.dec1, self.up1(x), skip1)

        return torch.sigmoid(self.head(x))