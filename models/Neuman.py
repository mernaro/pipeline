import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# Bloc Résiduel
# =========================
class ResidualBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        r = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + r)


# =========================
# Réseau de régularisation
# =========================
class RegularizationNetwork(nn.Module):
    def __init__(self, in_channels=1, n_residual_blocks=8, channels=64):
        super().__init__()

        self.entry = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(n_residual_blocks)]
        )

        self.exit = nn.Conv2d(channels, in_channels, 3, padding=1)

    def forward(self, x):
        x = self.entry(x)
        x = self.blocks(x)
        return self.exit(x)


# =========================
# Neumann Unrolling SR
# =========================
class NeumannSR(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.scale = getattr(config, "scale_factor", 2)
        self.max_iter = config.max_iterations
        self.tolerance = config.tolerance
        self.eta = config.eta
        self.in_channels = getattr(config, "in_channels", 1)

        self.reg_network = RegularizationNetwork(
            in_channels=self.in_channels,
            n_residual_blocks=config.n_residual_blocks,
            channels=config.feature_channels
        )

        # poids appris
        self.reg_weight = nn.Parameter(torch.tensor(0.1))


    # -------------------------
    # Normalisation entrée
    # -------------------------
    def _normalize_input(self, x):
        if x.dim() == 2:  # (H, W)
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            if x.shape[0] in [1, 3]:  # (C, H, W)
                x = x.unsqueeze(0)
            else:  # (H, W, C)
                x = x.permute(2, 0, 1).unsqueeze(0)
        return x


    # -------------------------
    # Opérateur A = S? S
    # -------------------------
    def A(self, x, lr_size):
        x_lr = F.interpolate(
            x, size=lr_size, mode="bicubic", align_corners=False
        )
        x_hr = F.interpolate(
            x_lr, size=(x.shape[2], x.shape[3]),
            mode="bicubic", align_corners=False
        )
        return x_hr


    # -------------------------
    # Forward
    # -------------------------
    def forward(self, lr, target_size=None):
      lr = self._normalize_input(lr)
      lr_size = (lr.shape[2], lr.shape[3])
  
      # initialisation bicubique
      if target_size is None:
          x = F.interpolate(lr, scale_factor=self.scale, mode="bicubic", align_corners=False)
          target_size = (x.shape[2], x.shape[3])
      else:
          x = F.interpolate(lr, size=(target_size[0]//self.scale, target_size[1]//self.scale),
                            mode="bicubic", align_corners=False)
  
      x0 = x.detach()
      alpha = F.softplus(self.reg_weight)
  
      for _ in range(self.max_iter):
          Ax = self.A(x, lr_size)
          data_term = x - self.eta * (Ax - x0)
          reg_term = self.reg_network(x)
          x = data_term + alpha * reg_term
  
      # forcer la sortie à la taille de target
      x = F.interpolate(x, size=target_size, mode="bicubic", align_corners=False)
      return torch.clamp(x, 0.0, 1.0)



    # -------------------------
    # Chargement depuis config
    # -------------------------
    @classmethod
    def from_config(cls, config):
        params = config["model"]["params"]

        cfg = type(
            "Config",
            (),
            {
                "scale_factor": params["scale_factor"],
                "max_iterations": params["max_iterations"],
                "tolerance": params["tolerance"],
                "eta": params["eta"],
                "n_residual_blocks": params["n_residual_blocks"],
                "feature_channels": params["feature_channels"],
                "in_channels": params["in_channels"],
            },
        )()

        model = cls(cfg)

        device = config.get(
            "device",
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        return model.to(device)
