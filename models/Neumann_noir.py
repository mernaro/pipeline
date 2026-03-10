import torch
import torch.nn as nn
import src.utils.Utils as Utils
from models.LearnedComponent import LearnedComponent

class NeumannSR(nn.Module):
    """
    Neumann unfolding with decimation:
        x_{k+1} = x_k - eta S? S x_k - R(x_k)
    """

    def __init__(self, learned_component: nn.Module, nb_blocks: int, eta: float, eta_learnable: bool):
        super().__init__()
        self.R = learned_component
        self.nb_blocks = nb_blocks

        if eta_learnable:
            self.eta = nn.Parameter(torch.tensor(float(eta)))
        else:
            self.register_buffer("eta", torch.tensor(float(eta)))

    def forward(self, low_resolution: torch.Tensor, decim_row: int, decim_col: int) -> torch.Tensor:
        """
        low_resolution : (B, C, h, w)
        decim_row / decim_col : upscaling factor
        """
        assert low_resolution.dim() == 4, "Input must be (B, C, h, w)"
        assert decim_row >= 1 and decim_col >= 1

        # Upsample low-res
        STy = Utils.decimation_adjoint_v3(low_resolution, decim_row, decim_col)

        # Clamp eta for stability
        eta = torch.clamp(self.eta, 1e-4, 1.0)

        current = eta * STy
        neumann_sum = current.clone()

        for _ in range(self.nb_blocks):
            # Linear term: A?A x
            decimated = Utils.decimation_v3(current, decim_row, decim_col)
            ATA_current = Utils.decimation_adjoint_v3(decimated, decim_row, decim_col)
            ATA_current = ATA_current / (decim_row * decim_col)

            linear_term = current - eta * ATA_current

            # Learned regularization
            R_out = self.R.network(current, is_training=self.training, n_residual_blocks=2)

            # Neumann update
            current = linear_term - R_out
            neumann_sum = neumann_sum + current

        # Normalisation finale pour avoir valeurs entre 0 et 1
        mini = neumann_sum.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
        maxi = neumann_sum.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
        normalized = (neumann_sum - mini) / (maxi - mini + 1e-8)

        return normalized

    @classmethod
    def from_config(cls, config: dict) -> "NeumannSR":
        params = config["model"]["params"]

        learned_component = LearnedComponent(
            in_channels=1,
            hidden_channels=params.get("nb_intermediate_channels", 64),
            kernel_size=params.get("kernel_size", 3),
            init_variance=0.05
        )

        model = cls(
            learned_component=learned_component,
            nb_blocks=params.get("nb_blocks", 4),
            eta=params.get("eta", 0.2),
            eta_learnable=params.get("eta_learnable", True)
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        return model.to(device)
