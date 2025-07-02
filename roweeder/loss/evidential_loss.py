import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module

def get_kl_divergence_prior_loss(alpha: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    alpha_kl = targets + (1 - targets) * alpha
    alpha_kl_sum = torch.sum(alpha_kl, dim=1, keepdim=True)
    ones = torch.ones_like(alpha)
    kl_log_term = (
        torch.lgamma(alpha_kl_sum)
        - torch.lgamma(torch.sum(ones, dim=1, keepdim=True))
        - torch.sum(torch.lgamma(alpha_kl), dim=1, keepdim=True)
    )
    kl_digamma_term = torch.sum(
        (alpha_kl - 1) * (torch.digamma(alpha_kl) - torch.digamma(alpha_kl_sum)), dim=1, keepdim=True
    )
    return (kl_log_term + kl_digamma_term).squeeze(dim=1)


class CrossEntropyBayesRiskLoss(Module):
    def __init__(self, kl_div_coeff: float = 0.1, ignore_index: int = -100, weights=[1.0, 1.0, 1.0], **kwargs):
        super().__init__()
        self.kl_div_coeff = kl_div_coeff
        self.ignore_index = ignore_index # Use a standard ignore_index
        self.weights = weights

    def forward(self, evidence: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
        # --- MODIFICATION: Adapt to RoWeeder's target format ---
        # RoWeeder target is [B, H, W] with integer class IDs.
        # This loss expects a one-hot encoded target of shape [B, C, H, W].
        
        num_classes = evidence.shape[1]
        
        # 1. Create the ignore mask from the integer target
        msk = (target != self.ignore_index).float()
        
        # 2. Create the one-hot encoded target
        targets_one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()

        # 3. Create the weights mask
        device = target.device
        weights_mask = torch.zeros_like(target, dtype=torch.float, device=device)
        for i, w in enumerate(self.weights):
            weights_mask[target == i] = w

        # --- Original UnsemLabAG loss logic ---
        alpha = evidence + 1
        S = torch.sum(alpha, dim=1, keepdim=True)
        xentropy_bayes_risk_loss = torch.sum(targets_one_hot * (torch.digamma(S) - torch.digamma(alpha)), dim=1)

        kl_div_prior_loss = get_kl_divergence_prior_loss(alpha, targets_one_hot)
        
        # Apply the ignore mask and weights
        loss = (xentropy_bayes_risk_loss + self.kl_div_coeff * kl_div_prior_loss) * msk
        loss = (loss * weights_mask).sum() / (msk.sum() + 1e-8) # Add epsilon for safety
        
        return loss