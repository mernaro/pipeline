import sys
sys.path.append('/projects/memaro/rpujol/unfolding/')
import numpy as np
import cv2
import lasp.metrics
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from pathlib import Path
import json
import torch

from models.Unfolding import Unfolding


class UnfoldingUNetPipeline:
    def __init__(self, 
                 unfolding_model_path,
                 config_path,
                 unet_model_path,
                 device='cuda'):
                 
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.unfolding_model = Unfolding.from_config(self.config)
        state_dict = torch.load(unfolding_model_path, map_location=torch.device(self.device))
        self.unfolding_model.load_state_dict(state_dict)
        self.unfolding_model.eval()
        
        self.unet = tf.keras.models.load_model(unet_model_path)
        
        print(f"Pipeline initialisee sur: {self.device}")


    def normalize_image(self, img):
        mini = np.min(img)
        maxi = np.max(img)
        if mini >= 0.0 and maxi <= 1.0:
            return img
        return (img - mini) / (maxi - mini)


    def compute_metrics(self, original, output):
        res_size = original.shape
        inp_size = output.shape
        img_low_res = output
        if res_size != inp_size:
            decim_row = res_size[0] // inp_size[0]
            decim_col = res_size[1] // inp_size[1]
            nb_row, nb_col = output.shape
            img_low_res = np.zeros((decim_row * nb_row, decim_col * nb_col), dtype=output.dtype)
            img_low_res[::decim_row, ::decim_col] = output.copy()
        psnr  = lasp.metrics.PSNR(original, img_low_res, intensity_max=1)
        mse   = lasp.metrics.MSE(original, img_low_res)
        mae   = lasp.metrics.MAE(original, img_low_res)
        maxae = np.max(np.abs(original - img_low_res))
        return psnr, mse, mae, maxae


    def preprocess_for_unfolding(self, image):
        if not isinstance(image, np.ndarray):
            raise TypeError(f"np.ndarray attendu, fourni: {type(image)}")
        
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif len(image.shape) != 2:
            raise ValueError(f"Shape image inattendue: {image.shape}")
        
        image = image.astype('float32')
        image = self.normalize_image(image)
        image_tensor = torch.from_numpy(image).float().to(self.device)
        
        return image_tensor


    def preprocess_for_unet(self, image):
         if len(image.shape) == 3:
             if image.shape[2] == 3:
                 image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
             elif image.shape[2] == 1:
                 image = image[:, :, 0]
         image = image.astype('float32')
         image = self.normalize_image(image)
         image = np.expand_dims(image, axis=-1)
         image = np.expand_dims(image, axis=0)
         return image


    def apply_super_resolution(self, low_res_image, high_res_size):
        image_tensor = self.preprocess_for_unfolding(low_res_image)
        
        inp_size = image_tensor.shape
        decim_row = high_res_size[0] // inp_size[0]
        decim_col = high_res_size[1] // inp_size[1]
        print(f"Facteurs de decimation: decim_row={decim_row}, decim_col={decim_col}")
        
        with torch.no_grad():
            hr_tensor = self.unfolding_model(image_tensor, decim_row, decim_col)
        
        print(f"Shape sortie Unfolding: {hr_tensor.shape}")
        
       
        hr_image = hr_tensor.squeeze().cpu().numpy()
        
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        
        return hr_image


    def load_image(self, image_path):
        if image_path.endswith('.npy'):
            image = np.load(image_path).astype('float32')
        else:
            image_bgr = cv2.imread(image_path)
            if image_bgr is None:
                raise FileNotFoundError(f"Image non trouvee: {image_path}")
            image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype('float32')
        return image


    def process_image(self, low_res_path, high_res_path):

        low_res = self.load_image(low_res_path)
        high_res_ref = self.load_image(high_res_path)
        
        low_res_size = low_res.shape[:2]
        high_res_size = high_res_ref.shape[:2]
        print(f"Taille Image Low Resolution: {low_res_size}")
        print(f"Taille Image High Resolution cible: {high_res_size}")
        
        # Sortie deja dans [0,1]
        high_res = self.apply_super_resolution(low_res, high_res_size)
        print(f"Taille Image Super Resolution produite: {high_res.shape[:2]}")

        low_res_normalized      = self.normalize_image(low_res.astype('float32'))
        high_res_normalized     = self.normalize_image(high_res.astype('float32'))
        high_res_ref_normalized = self.normalize_image(high_res_ref.astype('float32'))

        unet_input = self.preprocess_for_unet(high_res)
        unet_pred = self.unet.predict(unet_input, verbose=0)
        egmentation_mask = (unet_pred[0, :, :, 0] > 0.5).astype(np.uint8)

        # Pour sauvegarde png on remet dans [0,255]
        high_res_uint8 = (high_res * 255).astype(np.uint8)
        if len(high_res_uint8.shape) == 2:
            high_res_display = np.stack([high_res_uint8] * 3, axis=2)
        else:
            high_res_display = high_res_uint8

        overlay = high_res_display.copy()
        overlay[segmentation_mask > 0] = [255, 0, 0]

        results = {
            'low_res_normalized': low_res_normalized,
            'high_res_normalized': high_res_normalized,
            'high_res_ref_normalized': high_res_ref_normalized,
            'high_res_display': high_res_display,
            'segmentation_mask': segmentation_mask,
            'overlay': overlay,
            'metadata': {
                'low_res_shape': low_res_size,
                'high_res_shape': high_res_size,
                'decim_row': high_res_size[0] // low_res_size[0],
                'decim_col': high_res_size[1] // low_res_size[1]
            }
        }
        
        return results


    def visualize_results(self, results, output_dir, id_img):
        original         = results['high_res_ref_normalized']
        input_normalized = results['low_res_normalized']
        output           = results['high_res_normalized']

        psnr_input,  mse_input,  mae_input,  maxae_input  = self.compute_metrics(original, input_normalized)
        psnr_output, mse_output, mae_output, maxae_output = self.compute_metrics(original, output)

        title_full = (
            f"Mumford-Shah\n"
            f"Input (Low-Res) : PSNR: {psnr_input:.2f} | MSE: {mse_input:.4f} | MAE: {mae_input:.4f} | MaxAE: {maxae_input:.2f}\n"
            f"Reconstruction (High-Res) : PSNR: {psnr_output:.2f} | MSE: {mse_output:.4f} | MAE: {mae_output:.4f} | MaxAE: {maxae_output:.2f}"
        )

        fig = plt.figure(figsize=(10, 5))
        fig.subplots_adjust(wspace=0.05, hspace=0.1)
        fig.suptitle(title_full, fontsize=16, y=1.02)

        axes = fig.subplots(1, 3)
        images = [
            (axes[0], original,         "Original"),
            (axes[1], input_normalized, "Input (Low-Res)"),
            (axes[2], output,           "Reconstruction (High-Res)")
        ]
        for ax, img, title in images:
            ax.set_title(title, fontsize=14)
            ax.imshow(img, cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])

        save_path = os.path.join(output_dir, f"{id_img}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Figure sauvegardee dans: {save_path}")

        return psnr_input, mse_input, mae_input, maxae_input, psnr_output, mse_output, mae_output, maxae_output


    def save_results(self, results, output_dir="outputs"):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(
            f"{output_dir}/high_res.png",
            cv2.cvtColor(results['high_res_display'], cv2.COLOR_RGB2BGR)
        )
        print(f"Resultats sauvegardes dans: {output_dir}/")


if __name__ == "__main__":

    pipeline = UnfoldingUNetPipeline(
        unfolding_model_path="/projects/memaro/rpujol/unfolding/experiments/2026-02-06_14-18/best_model.pth",
        config_path="/projects/memaro/rpujol/unfolding/experiments/2026-02-06_14-18/config_used.json",
        unet_model_path="/projects/memaro/mcodjo/unfolding/models/unet_epoch_50.keras",
        device='cuda'
    )

    low_res_dir  = Path("/projects/memaro/rpujol/unfolding/data/dataset1/test/input/")
    high_res_dir = Path("/projects/memaro/rpujol/unfolding/data/dataset1/test/ground_truth/")
    output_dir   = Path("/projects/memaro/mcodjo/unfolding/experiments")

    low_res_paths = sorted(low_res_dir.glob("*.npy"))
    print(f"{len(low_res_paths)} image(s) trouvee(s)\n")

    for i, low_res_path in enumerate(low_res_paths):
        high_res_path = high_res_dir / low_res_path.name
        print(f"--- [{i+1}/{len(low_res_paths)}] Traitement de: {low_res_path.name} ---")

        try:
            results = pipeline.process_image(
                low_res_path=str(low_res_path),
                high_res_path=str(high_res_path)
            )

            image_output_dir = output_dir / low_res_path.stem
            image_output_dir.mkdir(parents=True, exist_ok=True)

            pipeline.save_results(results, output_dir=str(image_output_dir))
            pipeline.visualize_results(results, output_dir=str(image_output_dir), id_img=low_res_path.stem)

            print(f"Termine: {low_res_path.name}\n")

        except Exception as e:
            print(f"Erreur sur {low_res_path.name}: {e}\n")
            continue

    print("Traitement termine pour toutes les images.")