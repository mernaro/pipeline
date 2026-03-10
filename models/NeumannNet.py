import torch
import torch.nn as nn
import src.utils.Utils as Utils


class ResidualBlock(nn.Module):
    def __init__(self, n_channels: int = 1, n_blocks: int = 2):
        super().__init__()
        layers = []
        for _ in range(n_blocks):
            layers += [
                nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(n_channels),
                nn.ReLU(inplace=True),
            ]
        self.body = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class NeumannNet(nn.Module):
    def __init__(
        self,
        nb_iteration: int = 5,
        n_channels: int = 1,
        n_residual_blocks: int = 2,
        eta_init: float = 0.1,
    ):
        super().__init__()
        self.num_iters = nb_iteration
        self.eta = nn.Parameter(torch.tensor(eta_init))
        self.regularizers = nn.ModuleList(
            [ResidualBlock(n_channels, n_residual_blocks) for _ in range(nb_iteration)]
        )

    @classmethod
    def from_config(cls, config: dict) -> "NeumannNet":
        p = config["model"]["params"]
        return cls(
            nb_iteration       = p["nb_iteration"],
            n_channels         = p.get("n_channels", 1),
            n_residual_blocks  = p.get("n_residual_blocks", 2),
            eta_init           = p["beta0"]["initialize"],
        )

    def forward(
        self,
        y: torch.Tensor,
        decim_row: int,
        decim_col: int,
        sigma: float = 0.0,
    ) -> torch.Tensor:
        if y.dim() == 3:
            y = y.squeeze(0)

        Aty = Utils.decimation_adjoint(y, decim_row, decim_col)

        x0          = self.eta * Aty
        runner      = x0
        neumann_sum = runner

        for ii in range(self.num_iters):
            Ax      = Utils.decimation(runner, decim_row, decim_col)
            AtAx    = Utils.decimation_adjoint(Ax, decim_row, decim_col)

            linear_component  = runner - self.eta * AtAx
            learned_component = -self.regularizers[ii](runner.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)

            runner      = linear_component + learned_component
            neumann_sum = neumann_sum + runner
            
        #Normalisation
        mini = torch.min(neumann_sum)
        maxi = torch.max(neumann_sum)
        neumann_sum = (neumann_sum - mini) / (maxi - mini)

        return neumann_sum

    def get_metrics(self) -> dict:
        return {"eta": [self.eta.item()]}