import os
from PIL import Image
import numpy as np
import torch
from .utils import DataDict

class PhenoBenchDataset(torch.utils.data.Dataset):
    """
    Loads images and REAL ground truth from the PhenoBench dataset.
    Used for VALIDATION and TESTING.
    """
    id2class = {
        0: "background",
        1: "crop",
        2: "weed",
    }
    def __init__(self, root, transform=None, target_transform=None, **kwargs):
        self.split = kwargs.get("split", "val")
        self.root_dir = root
        self.transform = transform
        self.target_transform = target_transform

        self.image_dir = os.path.join(self.root_dir, self.split, 'images')
        self.label_dir = os.path.join(self.root_dir, self.split, 'semantics')
        self.image_files = sorted(os.listdir(self.image_dir))
        
        print(f"--- Initializing PhenoBenchDataset (Real GT) for split '{self.split}'. Found {len(self.image_files)} images. ---")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        target_pil = Image.open(label_path)

        if self.transform:
            image = self.transform(image)
        
        # Convert PIL mask to NumPy array for processing
        target_np = np.array(target_pil)

        # If mask is 3-channel (color-coded), convert to 2D label map
        if target_np.ndim == 3:
            target_np = np.argmax(target_np, axis=2)

        # Remap original PhenoBench class IDs to RoWeeder's 3-class format
        remapped_mask = np.zeros_like(target_np, dtype=np.int64)
        crop_mask = (target_np == 1) | (target_np == 3) | (target_np == 4)
        weed_mask = (target_np == 2)
        
        remapped_mask[crop_mask] = 1 # Crop class
        remapped_mask[weed_mask] = 2 # Weed class
        
        target = torch.from_numpy(remapped_mask)

        if self.target_transform:
            target = self.target_transform(target)
            
        assert target.ndim == 2, f"Target mask for {img_name} must be 2D, but was {target.shape}"
            
        return DataDict(image=image, target=target, name=os.path.join(self.split, img_name))


class SelfSupervisedPhenoBenchDataset(torch.utils.data.Dataset):
    """
    Loads original PhenoBench images but pairs them with generated PSEUDO-GT labels.
    Used for TRAINING ONLY.
    """

    id2class = {
        0: "background",
        1: "crop",
        2: "weed",
    }
    def __init__(self, root, gt_folder, transform=None, target_transform=None, **kwargs):
        self.split = kwargs.get("split", "train")
        self.root_dir = root
        self.gt_folder = gt_folder
        self.transform = transform
        self.target_transform = target_transform

        # Images are from the original PhenoBench dataset
        self.image_dir = os.path.join(self.root_dir, self.split, 'images')
        # Labels are from the folder where pseudo-GT was generated
        self.label_dir = os.path.join(self.gt_folder)

        self.image_files = sorted(os.listdir(self.image_dir))
        
        print(f"--- Initializing SelfSupervisedPhenoBenchDataset (Pseudo-GT). Using labels from: {self.label_dir} ---")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        pseudo_gt_pil = Image.open(label_path)
        
        if self.transform:
            image = self.transform(image)
        
        # Convert PIL mask to NumPy array
        target_np = np.array(pseudo_gt_pil, dtype=np.int64)
        
        # If the pseudo-GT was saved as a color image, convert it to a 2D label map
        if target_np.ndim == 3:
            # Assumes the brightest channel corresponds to the class
            target_np = np.argmax(target_np, axis=2)

        target = torch.from_numpy(target_np)

        if self.target_transform:
            target = self.target_transform(target)
            
        assert target.ndim == 2, f"Target mask for {img_name} must be 2D, but was {target.shape}"

        return DataDict(image=image, target=target, name=os.path.join(self.split, img_name))