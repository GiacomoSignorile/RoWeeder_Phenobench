import itertools
import os
import torch
import torchvision
import cv2
from torch.utils.data import Dataset
import torchvision
from roweeder.data.utils import DataDict, extract_plants, LABELS, pad_patches
from torchvision.transforms.functional import to_pil_image
from torchvision import transforms as T
class PhenoBenchDataset(Dataset):
    id2class = {
        0: "background",
        1: "crop",
        2: "weed",
    }
    def __init__(
        self,
        root,
        channels,
        fields,
        gt_folder=None,
        transform=None,
        target_transform=None,
        return_path=False,
        return_ndvi=False,
    ):
        super().__init__()
        self.root = root
        self.channels = channels
        # Set transform to identity if None
        self.transform = transform 
        # Set target_transform to identity if None
        self.target_transform = target_transform 
        self.return_path = return_path
        self.fields = fields
        self.return_ndvi = return_ndvi

        if gt_folder is None:
            self.gt_folders = {
                field: os.path.join(self.root, field, "groundtruth")
                for field in self.fields
            }
        else:
            self.gt_folders = {
                field: os.path.join(gt_folder, field) for field in self.fields
            }
            for k, v in self.gt_folders.items():
                if os.path.isdir(os.path.join(v, os.listdir(v)[0])):
                    self.gt_folders[k] = os.path.join(v, "groundtruth") 
        print(f"--- Using groundtruth folder: {self.gt_folders} ---")
        self.index = [
            (field, filename) for field in self.fields for filename in sorted(os.listdir(os.path.join(self.root, field, "images")))
        ]
        print(f"--- Initializing in PhenoBench mode for splits: {self.fields}. Found {len(self.index)} images. ---")


    def __len__(self):
        return len(self.index)
    
    def _get_gt(self, gt_path):
        gt = torchvision.io.read_image(gt_path)
        #print(f"[DEBUG] Groundtruth shape before processing: {gt.shape}")
        gt = gt[[2, 1, 0], ::]
        gt = gt.argmax(dim=0)
        #print(f"[DEBUG] Groundtruth shape after processing: {gt.shape}, unique values: {torch.unique(gt)}")
        if self.target_transform is not None:
            gt = self.target_transform(gt)
        return gt
    
    def _get_image(self, field, filename):
        channels = []
        #print(f"[DEBUG] Loading image for field: {field}, filename: {filename}, channels: {self.channels}")
        for channel_folder in self.channels:
            channel_path = os.path.join(
                self.root,
                field,
                channel_folder,
                filename
            )
            channel = torchvision.io.read_image(channel_path)
            channels.append(channel)
        channels = torch.cat(channels).float()  
        return self.transform(channels) 

    def _get_ndvi(self, field, filename):
        return "dummy_ndvi"  

    def __getitem__(self, i):
        field, filename = self.index[i]
        gt_path = os.path.join(
            self.gt_folders[field], filename
        )
        #print(f"[DEBUG] Groundtruth path: {gt_path}")
        gt = self._get_gt(gt_path)
        #print(f"[DEBUG] Groundtruth shape: {gt.shape}, unique values: {torch.unique(gt)}")
        channels = self._get_image(field, filename)
        #print(f"[DEBUG] Image {channels}")

        data_dict = DataDict(
            image = channels,
            target = gt,
        )
        if self.return_path:
            data_dict.name = gt_path
        
        if self.return_ndvi:
            ndvi = self._get_ndvi(field, filename)
            data_dict.ndvi = ndvi

        if self.transform is None:
            # If no transform is passed, apply a default one
            self.transform = T.Compose([T.ToTensor()])
        return data_dict


class SelfSupervisedPhenoBenchDataset(PhenoBenchDataset):
    def __init__(self, root, channels, fields, gt_folder=None, transform=None, target_transform=None, return_path=False, max_plants=10):
        super().__init__(root, channels, fields, gt_folder, transform, target_transform, return_path)
        self.max_plants = max_plants

    def __getitem__(self, i):
        data_dict = super().__getitem__(i)

        # Calculate connected components from groundtruth
        connected_components = torch.tensor(cv2.connectedComponents(
            data_dict.target.numpy().astype('uint8')
        )[1])

        # Process pseudo GT for crops
        crops_mask = data_dict.target == LABELS.CROP.value
        crops_mask = connected_components * crops_mask
        crops_mask = extract_plants(data_dict.image, crops_mask)
        if crops_mask.shape[0] > self.max_plants:
            indices = torch.randperm(crops_mask.shape[0])[:self.max_plants]
            crops_mask = crops_mask[indices]

        # Process pseudo GT for weeds
        weeds_mask = data_dict.target == LABELS.WEED.value
        weeds_mask = connected_components * weeds_mask
        weeds_mask = extract_plants(data_dict.image, weeds_mask)
        if weeds_mask.shape[0] > self.max_plants:
            indices = torch.randperm(weeds_mask.shape[0])[:self.max_plants]
            weeds_mask = weeds_mask[indices]

        # # Debug: Save preview image for pseudo-GT, if available
        # import os
        # import matplotlib.pyplot as plt
        # preview_folder = "pseudo_gt_previews"
        # os.makedirs(preview_folder, exist_ok=True)

        # if crops_mask.shape[0] > 0:
        #     crop_preview = crops_mask[0].cpu().numpy()
        #     print(f"Sample {i} crops_mask unique values: {torch.unique(crops_mask)}")
        #     print(f"Sample {i} crop_preview raw shape: {crop_preview.shape}, min: {crop_preview.min()}, max: {crop_preview.max()}")
        #     # If in CHW order with 3 channels, transpose to HWC
        #     if crop_preview.ndim == 3 and crop_preview.shape[0] == 3:
        #         crop_preview = crop_preview.transpose(1, 2, 0)
        #     plt.figure(figsize=(4, 4))
        #     plt.imshow(crop_preview, cmap='viridis', vmin=crop_preview.min(), vmax=crop_preview.max())
        #     plt.title(f"Sample {i} Crop Preview")
        #     preview_path = os.path.join(preview_folder, f"sample_{i}_crop.png")
        #     plt.savefig(preview_path)
        #     plt.close()
        #     print(f"Saved pseudo-GT preview for sample {i} at {preview_path}")
        # else:
        #     print(f"Sample {i} has no crops_mask (empty).")

        data_dict.crops = crops_mask
        data_dict.weeds = weeds_mask
        return data_dict

    def collate_fn(self, batch):
        crops = [item.crops for item in batch]
        weeds = [item.weeds for item in batch]
        crops = pad_patches(crops)
        weeds = pad_patches(weeds)
        return DataDict(
            image=torch.stack([item.image for item in batch]),
            target=torch.stack([item.target for item in batch]),
            crops=crops,
            weeds=weeds,
        )


class ClassificationPhenoBenchDataset(Dataset):
    id2class = {
        0: "crop",
        1: "weed",
    }
    def __init__(
        self,
        root,
        channels,
        fields,
        transform=None,
        target_transform=None,
        return_path=False,
    ):
        super().__init__()
        self.root = root
        self.channels = channels
        self.transform = transform
        self.return_path = return_path
        self.target_transform = target_transform
        self.fields = fields

        self.channels = channels
            
        self.index = [
            (field, filename) for field in self.fields for filename in os.listdir(os.path.join(self.root, field, channels[0]))
        ]
        
    def _get_image(self, field, filename):
        channels = []
        for channel_folder in self.channels:
            channel_path = os.path.join(
                self.root,
                field,
                channel_folder,
                filename
            )
            channel = torchvision.io.read_image(channel_path)
            channels.append(channel)
        channels = torch.cat(channels).float()
        return self.transform(channels)
    
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, i):
        field, filename = self.index[i]
        channels = self._get_image(field, filename)
        fname = os.path.splitext(filename)[0]
        label = int(fname.split("_")[-1])
        data_dict = DataDict(
            image = channels,
            target = self.target_transform(torch.tensor(label))
        )
        if self.return_path:
            data_dict.name = os.path.join(self.root, field, filename)
        return data_dict