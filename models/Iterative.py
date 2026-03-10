import torch
from models.IterationClassic import IterationClassic
from src.utils.Utils import decimation_adjoint

class SplitBregmanClassic(torch.nn.Module):

    def __init__(
        self,
        nb_iter: int = 10,
        alpha: float = 0.8,
        beta0: float = 0.02,
        beta1: float = 0.01,
        sigma: float = 1.0,
        nb_intermediate_channels: int = 1,
        kernel_size: tuple = (3, 3),
        taylor_nb_iterations: int = 5,
        taylor_kernel_size: tuple = (3, 3)
    ):
        super(SplitBregmanClassic, self).__init__()
        self.nb_iter = nb_iter
        self.iterations = torch.nn.ModuleList([
            IterationClassic(
                nb_intermediate_channels=nb_intermediate_channels,
                kernel_size=kernel_size,
                alpha=alpha,
                beta0=beta0,
                beta1=beta1,
                sigma=sigma,
                taylor_nb_iterations=taylor_nb_iterations,
                taylor_kernel_size=taylor_kernel_size
            ) for _ in range(nb_iter)
        ])

    @torch.no_grad()
    def forward(self, low_res: torch.Tensor, decim_row: int, decim_col: int) -> torch.Tensor:
        """
        Applique SplitBregman sur l'image LR pour reconstruire HR.
        Supporte batch et canal variable.
        """
        device = low_res.device
    
        # Assurer 4D : (B,C,H,W)
        if low_res.dim() == 2:
            low_res = low_res.unsqueeze(0).unsqueeze(0)
        elif low_res.dim() == 3:
            low_res = low_res.unsqueeze(0)
        B, C, H, W = low_res.shape
    
        # Sortie
        recon_batch = torch.zeros((B, 1, H*decim_row, W*decim_col), device=device)
    
        for b in range(B):
            lr_img = low_res[b, 0] 
    
            # STg pour cette image
            STg = decimation_adjoint(lr_img, decim_row, decim_col)
    
            # Initialisation
            f = torch.zeros_like(STg, device=device)
            d_x = torch.zeros_like(f)
            d_y = torch.zeros_like(f)
            b_x = torch.zeros_like(f)
            b_y = torch.zeros_like(f)
    
            # SplitBregman iteratif
            for it in self.iterations:
                f, d_x, d_y, b_x, b_y = it(STg, decim_row, decim_col, d_x, d_y, b_x, b_y)
    
            recon_batch[b, 0] = f
    
        # Normalisation globale
        mini = recon_batch.min()
        maxi = recon_batch.max()
        recon_batch = (recon_batch - mini) / (maxi - mini)
    
        return recon_batch


    def get_metrics(self):
        list_metrics = [it.update_metrics() for it in self.iterations]
        final_metrics = {k : {i : list_metrics[i][k] for i in range(len(list_metrics))} for k in list_metrics[0]}
        return final_metrics
