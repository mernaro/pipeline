import torch
from torch.utils import data
import numpy as np
import os

class ImageDataset(data.Dataset):
    def __init__(self, num_instances, data_part, data_dir="..."):
        # path_original = os.path.join(data_dir, f"{data_part}/original")
        # path_lowres = os.path.join(data_dir, f"{data_part}/low_resolution")
        path_original = os.path.join(data_dir, f"ground_truth")
        path_lowres = os.path.join(data_dir, f"input")

        self.items: list[tuple[torch.Tensor, torch.Tensor]] = []
        
        # chargement des fichiers .npy
        for n in range(num_instances):
            file_O = os.path.join(path_original, f"{n}.npy")
            file_LR = os.path.join(path_lowres, f"{n}.npy")

            # chargement et conversion
            O = np.load(file_O).astype("float32")
            LR = np.load(file_LR).astype("float32")

            O = self.normalize_image(O)
            LR = self.normalize_image(LR)
            
            # reshape et conversion en Tensor
            self.items.append((torch.from_numpy(O),torch.from_numpy(LR)))

    def __getitem__(self, index):
        img_orig = self.items[index][0]
        img_lowres = self.items[index][1]
        return img_orig, img_lowres

    def __len__(self):
        return len(self.items)

    def get_shape(self, path):
        first_file = os.path.join(path, "0.npy")
        sample = np.load(first_file).astype("float32")
        if len(sample.shape) == 2:
            sample = sample[np.newaxis, :, :]
        return sample.shape

    def normalize_image(self, img):
        mini = np.min(img)
        maxi = np.max(img)
        normalized = (img - mini)/(maxi - mini)
        return normalized
        
def get_batch_with_variable_size_image(batch):
    imgs_input = []
    imgs_ground_truth = []
    #imgs_filename = []

    for elem in batch:
        imgs_input.append(elem[0])
        imgs_ground_truth.append(elem[1])
        #imgs_filename.append(elem[2])

    # Your custom processing here
    return imgs_input, imgs_ground_truth