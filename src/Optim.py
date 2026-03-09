import torch
from datasets.ImageDataset import ImageDataset
import os
import numpy as np
from pathlib import Path

def generer_masques(image_dir, output_dir, seuil=0.5):

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    fichiers = sorted(Path(image_dir).glob("*.npy"))
    print(f"  {len(fichiers)} fichiers trouves dans {image_dir}")
    
    for path in fichiers:
        arr = np.load(path).astype(np.float32)

        if path == fichiers[0]:
            print(f"  Shape: {arr.shape} | min={arr.min():.3f} | max={arr.max():.3f} | dtype={arr.dtype}")
        
        # Normalise si valeurs en 0255
        if arr.max() > 1.0:
            arr = arr / 255.0
            
        masque = (arr < seuil).astype(np.float32)
        
        np.save(os.path.join(output_dir, path.name), masque)
        
        pct = masque.mean() * 100
        print(f"  {path.name} ? {pct:.1f}% objets")

for split in ["train", "val", "test"]:
    image_dir  = f"/projects/memaro/rpujol/unfolding/data/dataset1/{split}/ground_truth"
    output_dir = f"/projects/memaro/rpujol/unfolding/data/dataset1/{split}/masques"
    print(f"\n=== {split.upper()} ===")
    generer_masques(image_dir, output_dir)