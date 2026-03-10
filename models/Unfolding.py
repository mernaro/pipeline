
import torch
import torch.nn

import src.utils.Utils as Utils

from models.Iteration import Iteration

class Unfolding(torch.nn.Module):

    def __init__(self, 
        nb_intermediate_channels: int,
        kernel_size: tuple,
        nb_iterations: int,
        alpha: float,
        beta0: float,
        beta1: float,
        sigma: float,
        alpha_learnable: bool,
        beta0_learnable: bool,
        beta1_learnable: bool,
        sigma_learnable: bool,
        taylor_nb_iterations: int,
        taylor_kernel_size: tuple,
        taylor_generic: bool
    ) -> None:
        
        super(Unfolding, self).__init__()
        
        params = [
            nb_intermediate_channels,
            kernel_size,
            alpha,
            beta0,
            beta1,
            sigma,
            alpha_learnable,
            beta0_learnable,
            beta1_learnable,
            sigma_learnable,
            taylor_nb_iterations,
            taylor_kernel_size,
            taylor_generic
        ]

        iters = [ Iteration(*params) for _ in range(0, nb_iterations) ]
        
        self.iterations = torch.nn.ModuleList(iters)


    def forward(self, 
        low_resolution: torch.Tensor,
        decim_row: int,
        decim_col: int
    ) -> torch.Tensor:

        """
        
            Params:
                - low_resolution : image low-resolution
                - decim_row : decimation on line
                - decim_col : decimation on col

            Return:
                Image high-resolution of size.
                If size of low_resolution is (N, M), Image high-resolution
                will be (N*decim_row, M*decim_col)

        """


        # Initialize static attribute / shared attribute
        g = low_resolution
        STg = Utils.decimation_adjoint(g, decim_row, decim_col)
        
        d_x = torch.zeros_like(STg)
        d_y = torch.zeros_like(STg)
        b_x = torch.zeros_like(STg)
        b_y = torch.zeros_like(STg)
        for iter_layer in self.iterations:
            f, d_x, d_y, b_x, b_y = iter_layer(STg, decim_row, decim_col, d_x, d_y, b_x, b_y)
        f_approx = f
        
        # Normalize f
        mini = torch.min(f_approx)
        maxi = torch.max(f_approx)
        normalized = (f_approx - mini) / (maxi - mini)
        return normalized

    @classmethod
    def from_config(cls, config: dict) -> 'Unfolding':

        model_config = config['model']
        params = model_config['params']

        nb_intermediate_channels = params['nb_intermediate_channels']
        kernel_size = params['kernel_size']
        nb_iterations = params['nb_iteration']

        alpha = params['alpha']['initialize']
        beta0 = params['beta0']['initialize']
        beta1 = params['beta1']['initialize']
        sigma = params['sigma']['initialize']

        alpha_learnable = params['alpha']['is_learnable']
        beta0_learnable = params['beta0']['is_learnable']
        beta1_learnable = params['beta1']['is_learnable']
        sigma_learnable = params['sigma']['is_learnable']

        taylor_nb_iterations = params['taylor']['nb_iteration']
        taylor_kernel_size = params['taylor']['kernel_size']
        taylor_generic = params['taylor_generic']

        params = [
            nb_intermediate_channels,
            kernel_size,
            nb_iterations,
            alpha,
            beta0,
            beta1,
            sigma,
            alpha_learnable,
            beta0_learnable,
            beta1_learnable,
            sigma_learnable,
            taylor_nb_iterations,
            taylor_kernel_size,
            taylor_generic
        ]

        model = Unfolding(*params)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        return model.to(device)

    def get_metrics(self):
        list_metrics = [self.iterations[i].update_metrics() for i in range(len(self.iterations))]
        final_metrics = {k : {i : list_metrics[i][k] for i in range(len(list_metrics)) } for k in list_metrics[0]}
        return final_metrics