import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import lasp.metrics
import numpy as np
import src.utils.Utils as Utils


def plot_metrics(metrics, output_dir):
    for key, values in metrics.items():
        plt.figure()
        if isinstance(values, dict):
            for subkey, series in values.items():
                series = np.asarray(series)
                if series.ndim != 1:
                    series = series.ravel()
                plt.plot(range(1, len(series) + 1), series, linestyle='-', label=f"block_{subkey}")
            plt.title(f"Évolution de {key} (par block)")
            plt.xlabel("Époque")
            plt.ylabel(key)
            plt.legend(title="Block", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize="small")
            plt.grid(True, linestyle='--', alpha=0.6)
        else:
            series = np.asarray(values)
            if series.size == 0:
                print(f"Avertissement: la métrique {key} est vide, skip.")
                plt.close()
                continue
            if series.ndim != 1:
                series = series.ravel()
            plt.plot(range(1, len(series) + 1), series, linestyle='-')
            plt.title(f"Évolution de {key}")
            plt.xlabel("Époque")
            plt.ylabel(key)
            plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        save_path = os.path.join(output_dir, f"{key}.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Figure sauvegardée : {save_path}")


def compute_metrics(original, output):
    res_size = original.shape
    inp_size = output.shape
    img_low_res = output
    if res_size != inp_size:
        decim_row = res_size[0] // inp_size[0]
        decim_col = res_size[1] // inp_size[1]
        nb_row, nb_col = output.shape
        img_low_res = np.zeros((decim_row * nb_row, decim_col * nb_col), dtype=output.dtype)
        img_low_res[::decim_row, ::decim_col] = output.copy()
    psnr = lasp.metrics.PSNR(original, img_low_res, intensity_max=1)
    mse  = lasp.metrics.MSE(original, img_low_res)
    mae  = lasp.metrics.MAE(original, img_low_res)
    return psnr, mse, mae



def show_and_save_4images(original, input_lr, sr_output, seg_mask, output_dir, id_img, params):
   
    def _squeeze(img):
        arr = np.array(img)
        while arr.ndim > 2:
            arr = arr.squeeze(0)
        return arr

    original  = _squeeze(original)
    input_lr  = _squeeze(input_lr)
    sr_output = _squeeze(sr_output)
    seg_mask  = _squeeze(seg_mask).astype(np.uint8)

    # Metriques SR
    psnr_in,  mse_in,  mae_in  = compute_metrics(original, input_lr)
    psnr_out, mse_out, mae_out = compute_metrics(original, sr_output)

    # Masque binaire inversé 
    seg_bw = 1.0 - seg_mask.astype(np.float32)


    title_full = (
        f"Pipeline SR + Segmentation\n"
        f"Blur: {params[0]}x{params[0]}, σ={params[1]:.1f} | "
        f"Decimation: {params[2]}x{params[2]} | SNRdB: {params[4]:.1f}\n"
        f"LR → PSNR: {psnr_in:.2f} | MSE: {mse_in:.4f} | MAE: {mae_in:.4f} \n"
        f"SR → PSNR: {psnr_out:.2f} | MSE: {mse_out:.4f} | MAE: {mae_out:.4f}"
    )

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle(title_full, fontsize=11, y=1.10)
    fig.subplots_adjust(wspace=0.05, top=0.78)

    columns = [
        (axes[0], original,       "GT (Original HR)",     "gray", None),
        (axes[1], input_lr,       "LR (Entrée)",          "gray", None),
        (axes[2], sr_output,      "SR (Reconstruction)",  "gray", None),
        (axes[3], seg_bw,         "Segmentation (B/N)",   "gray", None),
    ]

    for ax, img, title, cmap, _ in columns:
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])


    save_path = os.path.join(output_dir, f"{id_img}_eval.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()

    #plot_histogram_gray(sr_output, os.path.join(output_dir, f"{id_img}_hist.png"))

    return (
        psnr_in,  mse_in,  mae_in,
        psnr_out, mse_out, mae_out,
    )


def show_and_save_3images(original, input_normalized, output, output_dir, id_img, params):
    fig = plt.figure(figsize=(10, 5))
    fig.subplots_adjust(wspace=0.05, hspace=0.1)

    psnr_input,  mse_input,  mae_input  = compute_metrics(original, input_normalized)
    psnr_output, mse_output, mae_output = compute_metrics(original, output)

    title_full = (
        f"Unfolding\n"
        f"Blur filter: {params[0]}x{params[0]}, σ={params[1]} | "
        f"Decimation: {params[2]}x{params[2]} | SNRdB: {params[4]}\n"
        f"Input (Low-Res) : PSNR: {psnr_input:.2f} | MSE: {mse_input:.4f} | MAE: {mae_input:.4f}\n"
        f"Reconstruction (High-Res) : PSNR: {psnr_output:.2f} | MSE: {mse_output:.4f} | MAE: {mae_output:.4f}\n"
    )
    fig.suptitle(title_full, fontsize=16, y=1.02)

    axes = fig.subplots(1, 3)
    images = [
        (axes[0], original,          "Original"),
        (axes[1], input_normalized,  "Input (Low-Res)"),
        (axes[2], output,            "Reconstruction (High-Res)"),
    ]
    for ax, img, title in images:
        ax.set_title(title, fontsize=14)
        ax.imshow(img, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])

    save_path = os.path.join(output_dir, f"{id_img}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    #plot_histogram_gray(output, os.path.join(output_dir, f"{id_img}_hist.png"))

    return (
        psnr_input,  mse_input,  mae_input,
        psnr_output, mse_output, mae_output,
    )


def plot_histogram_gray(image, filename):
    if image.min() < 0 or image.max() > 1:
        mini = image.min()
        maxi = image.max()
        image = (image - mini) / (maxi - mini + 1e-8)

    plt.figure(figsize=(6, 4))
    plt.hist(image.flatten(), bins=50, range=(0, 1))
    plt.title("Histogramme des niveaux de gris")
    plt.xlabel("Valeur (0 = noir, 1 = blanc)")
    plt.ylabel("Nombre de pixels")
    plt.savefig(filename, dpi=300)
    plt.close()