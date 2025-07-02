# This is a standard, self-contained implementation of the Lovász-Softmax loss.
# Original source: https://github.com/bermanmaxim/LovaszSoftmax

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Module
from .utils import get_reduction

def lovasz_grad(gt_sorted):
    """Computes gradient of the Lovasz extension w.r.t sorted errors"""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

class LovaszSoftmax(Module):
    def __init__(self, reduction: str = "mean", label_smoothing: float = 0.2, **kwargs):
        super().__init__()
        self.reduction = get_reduction(reduction)
        self.label_smoothing = label_smoothing

    def __call__(self, x: torch.Tensor, target: torch.Tensor, weight_matrix=None, **kwargs) -> torch.Tensor:
        # x is the model's output logits: [Batch, Num_Classes, H, W]
        # target is the ground truth: [Batch, H, W]
        num_classes = x.shape[1]
        
        # Get probabilities from logits
        probas = F.softmax(x, dim=1)
        
        # If label smoothing is enabled, create a smoothed one-hot target
        if self.label_smoothing > 0:
            # print(f"[LovaszSoftmax] Using label smoothing: {self.label_smoothing}")
            with torch.no_grad():
                # target: [B, H, W] -> [B, num_classes, H, W] one-hot
                target_onehot = torch.zeros_like(x).scatter_(1, target.unsqueeze(1), 1)
                target_smooth = target_onehot * (1 - self.label_smoothing) + self.label_smoothing / num_classes
        else:
            target_smooth = None  # Not used if no smoothing
        
        # The Lovasz loss is calculated per class and then averaged
        loss = 0.0
        for c in range(num_classes):
            if self.label_smoothing > 0:
                fg = target_smooth[:, c].reshape(-1)
            else:
                fg = (target == c).float().reshape(-1)
            class_proba = probas[:, c].reshape(-1)
            
            errors = (fg - class_proba).abs()
            errors_sorted, perm = torch.sort(errors, 0, descending=True)
            perm = perm.data
            fg_sorted = fg[perm]
            
            # Calculate the Lovasz gradient and loss for this class
            grad = lovasz_grad(fg_sorted)
            class_loss = torch.dot(errors_sorted, grad)
            loss += class_loss
            
        # Average the loss across all classes
        final_loss = loss / num_classes
        
        # The original implementation does not easily support a weight_matrix in the same way.
        # Class weighting is often handled by weighting the `class_loss` before summing.
        # For now, we omit it for simplicity, as Lovasz is already good with imbalance.
        
        return final_loss # Reduction is implicitly 'mean' by averaging over classes