from copy import deepcopy
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import torchvision.transforms as T
from roweeder.data.phenobench import (PhenoBenchDataset, SelfSupervisedPhenoBenchDataset, ClassificationPhenoBenchDataset)
from roweeder.data.weedmap import (
    SelfSupervisedWeedMapDataset,
    WeedMapDataset,
    ClassificationWeedMapDataset,
)
from torch.utils.data import Subset



def get_preprocessing(dataset_params):
    preprocess_params = dataset_params.pop("preprocess")
    transforms = T.Compose(
        [
            T.ToTensor(),
            lambda x: x.float() / 255.0,
            T.Normalize(
                mean=torch.tensor(preprocess_params["mean"]),
                std=torch.tensor(preprocess_params["std"]),
            ),
        ]
    )
    if "resize" in preprocess_params:
        resize_val = preprocess_params["resize"]
        transforms.transforms.insert(0, T.Resize(resize_val, interpolation=T.InterpolationMode.BILINEAR))
        target_transforms = T.Compose([
            # If target is 2D, add a channel dimension:
            T.Lambda(lambda x: x.unsqueeze(0) if x.ndim == 2 else x),
            T.Resize(resize_val, interpolation=T.InterpolationMode.NEAREST),
            # Remove the dummy channel
            T.Lambda(lambda x: x.squeeze(0) if x.ndim == 3 else x),
            T.Lambda(lambda x: torch.as_tensor(x, dtype=torch.long))
        ])
    else:
        target_transforms = T.Lambda(lambda x: torch.as_tensor(np.array(x), dtype=torch.long))
    deprocess = T.Compose(
        [
            T.Normalize(
                mean=[-m / s for m, s in zip(preprocess_params["mean"], preprocess_params["std"])],
                std=[1 / s for s in preprocess_params["std"]],
            ),
            lambda x: x * 255.0,
        ]
    )
    return transforms, target_transforms, deprocess


def get_classification_dataloaders(dataset_params, dataloader_params, seed=42):
    dataset_params = deepcopy(dataset_params)
    transforms, target_transforms, deprocess = get_preprocessing(dataset_params)
    
    
    if "test_preprocess" in dataset_params:
        dataset_params["preprocess"] = dataset_params["test_preprocess"]
        dataset_params.pop("test_preprocess")
        test_transforms, test_target_transforms, test_deprocess = get_preprocessing(dataset_params)
    else:
        test_transforms = transforms
        test_target_transforms = target_transforms
        test_deprocess = deprocess

    if "train_fields" in dataset_params:
        train_params = deepcopy(dataset_params)
        train_params["fields"] = dataset_params["train_fields"]
        train_params.pop("train_fields")
        train_params.pop("test_fields")
        train_params.pop("test_root", None)
        train_set = ClassificationPhenoBenchDataset(
            **train_params,
            transform=transforms,
            target_transform=target_transforms,
        )
        val_set = ClassificationPhenoBenchDataset(
            **train_params,
            transform=transforms,
            target_transform=target_transforms,
        )
        # Debug: print a preview of the groundtruth for the validation set
        sample_val = val_set[0]
        print("Validation groundtruth preview (unique labels):", torch.unique(sample_val.target))
        print("Validation groundtruth preview (shape):", sample_val.target.shape)

        # Save preview image to a folder
        import os
        import matplotlib.pyplot as plt

        preview_folder = "gt_validation_previews"
        print("Current working directory:", os.getcwd())
        if not os.path.exists(preview_folder):
            os.makedirs(preview_folder, exist_ok=True)
        gt_np = sample_val.target.cpu().numpy()
        # If a dummy channel exists (shape [1, H, W]), squeeze it out for visualization
        if gt_np.ndim == 3 and gt_np.shape[0] == 1:
            gt_np = gt_np.squeeze(0)
        plt.figure(figsize=(6,6))
        plt.imshow(gt_np, cmap="gray")
        plt.title("Validation GT Preview")
        preview_path = os.path.join(preview_folder, "validation_gt_preview.png")
        plt.savefig(preview_path)
        plt.close()
        print("Saved validation groundtruth preview to:", preview_path)
        
        index = train_set.index
        train_index, val_index = train_test_split(
            index, test_size=0.2, random_state=seed
        )
        train_set.index = train_index
        val_set.index = val_index

        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=dataloader_params["batch_size"],
            shuffle=True,
            num_workers=dataloader_params["num_workers"],
        )
        val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=dataloader_params["batch_size"],
            shuffle=False,
            num_workers=dataloader_params["num_workers"],
        )
    else:
        train_loader = None
        val_loader = None
    dataset_params["return_ndvi"] = True
    test_loader = get_testloader(
        dataset_params,
        dataloader_params,
        test_transforms,
        target_transforms=test_target_transforms,
    )
    return train_loader, val_loader, test_loader, deprocess

def get_dataloaders(dataset_params, dataloader_params, seed=42):
    if "gt_folder" not in dataset_params:
        return get_classification_dataloaders(dataset_params, dataloader_params, seed)
        
    dataset_params = deepcopy(dataset_params)
    transforms, target_transforms, deprocess = get_preprocessing(dataset_params)

    if "train_fields" in dataset_params:
        train_params = deepcopy(dataset_params)
        train_params["fields"] = dataset_params["train_fields"]
        train_params.pop("train_fields")
        train_params.pop("test_fields")
        
        # Use SelfSupervisedPhenoBenchDataset for training and validation (i.e. pseudo-GT)
        train_set = SelfSupervisedPhenoBenchDataset(
            **train_params,
            transform=transforms,
            target_transform=target_transforms,
        )
        # For validation, also use the self-supervised dataset
        val_set = SelfSupervisedPhenoBenchDataset(
            **train_params,
            transform=transforms,
            target_transform=target_transforms,
        )
        
        index = train_set.index
        train_index, val_index = train_test_split(
            index, test_size=0.2, random_state=seed
        )
        train_set.index = train_index
        val_set.index = val_index

        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=dataloader_params["batch_size"],
            shuffle=True,
            num_workers=dataloader_params["num_workers"],
            collate_fn=train_set.collate_fn,
        )
        val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=dataloader_params["batch_size"],
            shuffle=False,
            num_workers=dataloader_params["num_workers"],
            collate_fn=val_set.collate_fn,
        )
    else:
        train_loader = None
        val_loader = None
        
    # Test loader still uses PhenoBenchDataset (true GT)
    test_loader = get_testloader(
        dataset_params, dataloader_params, transforms, target_transforms
    )

    return train_loader, val_loader, test_loader, deprocess


def get_testloader(dataset_params, dataloader_params, transforms, target_transforms):
    test_params = deepcopy(dataset_params)
    test_params["fields"] = dataset_params["test_fields"]
    test_params.pop("test_fields")
    test_params.pop("gt_folder", None)
    if "test_root" in dataset_params:
        test_params["root"] = dataset_params["test_root"]
        test_params.pop("test_root")
    if "train_fields" in dataset_params:
        test_params.pop("train_fields")

    test_set = PhenoBenchDataset(
        transform=transforms,
        target_transform=target_transforms,
        **test_params,
    )
    
    # -- Debug: print a preview of the groundtruth --
    sample = test_set[0]
    print("Groundtruth preview (unique labels):", torch.unique(sample.target))
    print("Groundtruth preview (shape):", sample.target.shape)
    # -----------------------------------------------------
    
    return torch.utils.data.DataLoader(
        test_set,
        batch_size=dataloader_params["batch_size"],
        shuffle=False,
        num_workers=dataloader_params["num_workers"],
    )


def get_dataset(root, modality, fields=None):
    if fields is None:
        fields = []
    if modality == "PhenoBench":
        # Use the PhenoBench dataset class, for example:
        split = "train"  # or choose based on extra parameters
        return PhenoBenchDataset(root_dir=root, split=split)
    else:
        # Original logic for the WeedMap-style dataset using fields
        return WeedMapDataset(root, fields)