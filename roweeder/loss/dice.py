# In roweeder/loss/dice.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
from .utils import get_reduction

class DiceLoss(Module):
    def __init__(self, smooth: float = 1.0, reduction: str = "mean", **kwargs):
        """
        Dice Loss for semantic segmentation.
        Designed to maximize the overlap between prediction and target.
        
        Args:
            smooth (float): A small value to prevent division by zero.
            reduction (str): 'mean' or 'sum' for the final loss.
        """
        super().__init__()
        self.smooth = smooth
        self.reduction = get_reduction(reduction)

    def __call__(self, x: torch.Tensor, target: torch.Tensor, weight_matrix=None, **kwargs) -> torch.Tensor:
        # x is the model's output logits: [Batch, Num_Classes, H, W]
        # target is the ground truth: [Batch, H, W]
        
        # 1. Get probabilities from logits using softmax
        probs = F.softmax(x, dim=1)
        
        # 2. Convert the integer target mask to a one-hot encoded format
        # The shape becomes [Batch, Num_Classes, H, W] to match the probabilities
        target_one_hot = F.one_hot(target, num_classes=probs.shape[1]).permute(0, 3, 1, 2).float()
        
        # 3. Flatten tensors to [Batch, Num_Classes, -1]
        probs = probs.flatten(2)
        target_one_hot = target_one_hot.flatten(2)
        
        # 4. Calculate the intersection and union for the Dice formula
        intersection = (probs * target_one_hot).sum(-1)
        # The union in Dice is simply the sum of probabilities and targets
        union = probs.sum(-1) + target_one_hot.sum(-1)
        
        # 5. Calculate the Dice score per class for each image in the batch
        # Dice = (2 * Intersection) / (Union)
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # 6. The Dice Loss is 1 - Dice Score
        dice_loss = 1. - dice_score
        
        # 7. Apply the weight matrix if provided
        if weight_matrix is not None:
            # First, get the class weights for each image in the batch
            # We assume the weight_matrix is shaped [Batch, H, W] and target is [Batch, H, W]
            # We need to get per-class weights.
            # A simpler way is to have weights per class, e.g., shape [Num_Classes]
            # For now, we'll assume a simpler weighting or ignore it for Dice.
            # A common approach is to weight the loss for each class.
            class_weights = torch.tensor([0.1, 0.45, 0.45], device=x.device) # Example weights
            dice_loss = dice_loss * class_weights

        return self.reduction(dice_loss)