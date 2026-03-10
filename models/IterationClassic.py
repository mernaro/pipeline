import torch
from models.Iteration import Iteration
import src.utils.Utils as Utils


class IterationClassic(Iteration):

    def __init__(
        self,
        nb_intermediate_channels: int,
        kernel_size: tuple,
        alpha: float,
        beta0: float,
        beta1: float,
        sigma: float,
        taylor_nb_iterations: int,
        taylor_kernel_size: tuple
    ):
        super().__init__(
            nb_intermediate_channels=nb_intermediate_channels,
            kernel_size=kernel_size,
            alpha=alpha,
            beta0=beta0,
            beta1=beta1,
            sigma=sigma,
            alpha_learnable=False,
            beta0_learnable=False,
            beta1_learnable=False,
            sigma_learnable=False,
            taylor_nb_iterations=taylor_nb_iterations,
            taylor_kernel_size=taylor_kernel_size
        )

        self.gt = None
        self.metrics.update({
            "psnr": [],
            "ssim": []
        })

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, STg, decim_row, decim_col, d_x, d_y, b_x, b_y):
        f, d_x, d_y, b_x, b_y = super().forward(
            STg, decim_row, decim_col, d_x, d_y, b_x, b_y
        )

        with torch.no_grad():
            if self.gt is not None:
                psnr_val = Utils.psnr(f, self.gt)
                ssim_val = Utils.ssim(f, self.gt)
                self.metrics["psnr"].append(psnr_val.item())
                self.metrics["ssim"].append(ssim_val.item())

        return f, d_x, d_y, b_x, b_y
