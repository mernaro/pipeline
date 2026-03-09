from src.utils.UtilsPlot import show_and_save_3images
import torch
import numpy as np
import pandas as pd
import os

def evaluation_splitbregman(model, evaluation_loader, output_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()


    psnr_input_list, mse_input_list, mae_input_list, maxae_input_list = [], [], [], []
    psnr_output_list, mse_output_list, mae_output_list, maxae_output_list = [], [], [], []

    with torch.no_grad():
        for _, (O, L, P) in enumerate(evaluation_loader):
            for i in range(len(O)):
                original_true = O[i].to(device)
                low_resolution = L[i].to(device)
                params = P[i].to(device)

                # Assurer 4 dimensions : (B,C,H,W)
                if low_resolution.dim() == 2:  # H,W
                    lr_img_tensor = low_resolution.unsqueeze(0).unsqueeze(0)
                elif low_resolution.dim() == 3:  # C,H,W
                    lr_img_tensor = low_resolution.unsqueeze(0)
                else:
                    lr_img_tensor = low_resolution

                if original_true.dim() == 2:
                    hr_img_tensor = original_true.unsqueeze(0).unsqueeze(0)
                elif original_true.dim() == 3:
                    hr_img_tensor = original_true.unsqueeze(0)
                else:
                    hr_img_tensor = original_true

                # Calcul dynamique du facteur de decimation
                decim_row = hr_img_tensor.shape[2] // lr_img_tensor.shape[2]
                decim_col = hr_img_tensor.shape[3] // lr_img_tensor.shape[3]

                # Reconstruction Split-Bregman
                reconstructed = model(lr_img_tensor.to(device), decim_row, decim_col)

                # Conversion numpy pour utilitaire d'affichage
                original_true_np = hr_img_tensor.cpu().numpy()
                low_resolution_np = lr_img_tensor.cpu().numpy()

                if reconstructed.dim() == 4:  # (B,C,H,W)
                    reconstructed_np = reconstructed.squeeze(0).squeeze(0).cpu().numpy()
                elif reconstructed.dim() == 3:  # (C,H,W)
                    reconstructed_np = reconstructed.squeeze(0).cpu().numpy()
                else:
                    reconstructed_np = reconstructed.cpu().numpy()

                # Calcul et affichage via ton utilitaire existant
                (
                    psnr_in, mse_in, mae_in, maxae_in,
                    psnr_out, mse_out, mae_out, maxae_out
                ) = show_and_save_3images(
                    original_true_np,
                    low_resolution_np,
                    reconstructed_np,
                    output_dir,
                    len(psnr_output_list),
                    params
                )

                # Stocker les metriques
                psnr_input_list.append(psnr_in)
                mse_input_list.append(mse_in)
                mae_input_list.append(mae_in)
                maxae_input_list.append(maxae_in)
                psnr_output_list.append(psnr_out)
                mse_output_list.append(mse_out)
                mae_output_list.append(mae_out)
                maxae_output_list.append(maxae_out)

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

    # Ajouter ligne moyenne
    df.loc[len(df)] = (
        "Moyenne",
        np.mean(psnr_input_list), np.mean(psnr_output_list),
        np.mean(mse_input_list), np.mean(mse_output_list),
        np.mean(mae_input_list), np.mean(mae_output_list),
        np.mean(maxae_input_list), np.mean(maxae_output_list)
    )

    df = df.round(3)
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "metrics_splitbregman.csv")
    df.to_csv(csv_path, index=False)

