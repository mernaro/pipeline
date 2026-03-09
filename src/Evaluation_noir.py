import torch
import numpy as np
import pandas as pd
import os
from src.utils.UtilsPlot import show_and_save_3images

def evaluation(model, evaluation_loader, output_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()

    # Stockage des métriques
    psnr_input_list, mse_input_list, mae_input_list, maxae_input_list = [], [], [], []
    psnr_output_list, mse_output_list, mae_output_list, maxae_output_list = [], [], [], []

    with torch.no_grad():
        for _, (O, L, P) in enumerate(evaluation_loader):
            for i in range(len(O)):
                original_true = O[i].to(device)
                low_resolution = L[i].to(device)
                params = P[i]

                # S'assurer que les images sont 4D (B, C, H, W)
                if original_true.dim() == 2:
                    original_true = original_true.unsqueeze(0).unsqueeze(0)
                elif original_true.dim() == 3:
                    original_true = original_true.unsqueeze(0)
                if low_resolution.dim() == 2:
                    low_resolution = low_resolution.unsqueeze(0).unsqueeze(0)
                elif low_resolution.dim() == 3:
                    low_resolution = low_resolution.unsqueeze(0)

                # Calcul des facteurs de décimation
                decim_row = original_true.shape[2] // low_resolution.shape[2]
                decim_col = original_true.shape[3] // low_resolution.shape[3]
                original_pred = model(low_resolution, decim_row, decim_col)

                # Convertir en numpy pour métriques et affichage
                def to_numpy(img):
                    if isinstance(img, torch.Tensor):
                        img = img.detach().cpu()
                    while img.ndim > 3:
                        img = img.squeeze(0)
                    return img.numpy()

                original_true_np = to_numpy(original_true)
                low_resolution_np = to_numpy(low_resolution)
                original_pred_np = to_numpy(original_pred)

                # Calcul et sauvegarde des métriques
                psnr_in, mse_in, mae_in, maxae_in, psnr_out, mse_out, mae_out, maxae_out = show_and_save_3images(
                    original_true_np,
                    low_resolution_np,
                    original_pred_np,
                    output_dir,
                    len(psnr_output_list),
                    params=params
                )

                psnr_input_list.append(psnr_in)
                mse_input_list.append(mse_in)
                mae_input_list.append(mae_in)
                maxae_input_list.append(maxae_in)
                psnr_output_list.append(psnr_out)
                mse_output_list.append(mse_out)
                mae_output_list.append(mae_out)
                maxae_output_list.append(maxae_out)

    # Sauvegarde CSV
    df = pd.DataFrame({
        "Image_ID": list(range(len(psnr_output_list))),
        "PSNR_Input": psnr_input_list,
        "PSNR_Output": psnr_output_list,
        "MSE_Input": mse_input_list,
        "MSE_Output": mse_output_list,
        "MAE_Input": mae_input_list,
        "MAE_Output": mae_output_list,
        "MaxAE_Input": maxae_input_list,
        "MaxAE_Output": maxae_output_list,
    })

    # Moyenne
    df.loc[len(df)] = (
        "Moyenne",
        np.mean(psnr_input_list), np.mean(psnr_output_list),
        np.mean(mse_input_list), np.mean(mse_output_list),
        np.mean(mae_input_list), np.mean(mae_output_list),
        np.mean(maxae_input_list), np.mean(maxae_output_list)
    )

    df = df.round(3)
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "metrics.csv")
    df.to_csv(csv_path, index=False)
