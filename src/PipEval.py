import os
import numpy as np
import pandas as pd
from pathlib import Path
import torch

from src.utils.UtilsPlot import show_and_save_4images


def evaluation(trainer, evaluation_loader, output_dir):

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    device = trainer.device
    trainer.unfolding.eval()
    trainer.unet.eval()

    psnr_input_list,  mse_input_list,  mae_input_list  = [], [], []
    psnr_output_list, mse_output_list, mae_output_list = [], [], []
    seg_pixels_list = []

    with torch.no_grad():
        for _, (O, L, P, S) in enumerate(evaluation_loader):
            for i in range(len(O)):
                original_true  = O[i].to(device)
                low_resolution = L[i].to(device)
                params         = P[i].to(device)

                res_size  = original_true.size()
                inp_size  = low_resolution.size()
                decim_row = res_size[0] // inp_size[0]
                decim_col = res_size[1] // inp_size[1]

                # Super-resolution
                original_pred = trainer.unfolding(low_resolution, decim_row, decim_col)

                # Segmentation
                seg_prob = trainer.unet(original_pred).squeeze().cpu().numpy()
                seg_mask = (seg_prob > 0.5).astype(np.uint8)

                img_id = len(psnr_output_list)

                # Visualisation 4 colonnes : GT | LR | SR | SR+seg
                (
                    psnr_in,  mse_in,  mae_in,
                    psnr_out, mse_out, mae_out,
                ) = show_and_save_4images(
                    original_true.cpu().numpy(),
                    low_resolution.cpu().numpy(),
                    original_pred.cpu().numpy(),
                    seg_mask,
                    output_dir,
                    img_id,
                    params.cpu().numpy(),
                )

                # Sauvegarde masque + proba brute
                np.save(os.path.join(output_dir, f"seg_mask_{img_id}.npy"), seg_mask)
                np.save(os.path.join(output_dir, f"seg_prob_{img_id}.npy"), seg_prob)

                psnr_input_list.append(psnr_in)
                mse_input_list.append(mse_in)
                mae_input_list.append(mae_in)
                psnr_output_list.append(psnr_out)
                mse_output_list.append(mse_out)
                mae_output_list.append(mae_out)
                seg_pixels_list.append(float(seg_mask.sum()) / seg_mask.size * 100)

    df = pd.DataFrame({
        "Image_ID":      list(range(len(psnr_output_list))),
        "PSNR_Input":    psnr_input_list,
        "PSNR_Output":   psnr_output_list,
        "MSE_Input":     mse_input_list,
        "MSE_Output":    mse_output_list,
        "MAE_Input":     mae_input_list,
        "MAE_Output":    mae_output_list,
        "Seg_pct":       seg_pixels_list,
    })

    df.loc[len(df)] = (
        "Moyenne",
        np.mean(psnr_input_list),  np.mean(psnr_output_list),
        np.mean(mse_input_list),   np.mean(mse_output_list),
        np.mean(mae_input_list),   np.mean(mae_output_list),
        np.mean(seg_pixels_list),
    )

    df = df.round(3)
    csv_path = os.path.join(output_dir, "metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nFichier de metriques sauvegarde : {csv_path}")