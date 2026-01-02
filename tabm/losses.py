# Contains pytorch losses for ensemble models. 
# Training loss will be invidual for each memeber of the ensemble
# Validation/Testing/Inference loss will use averaged output
# Average for classification = average probabilities (not logits)

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

# ------------------------------
# Regression Loss
# ------------------------------

def regression_loss(
    y_pred: torch.Tensor, 
    y_true: torch.Tensor, 
    training: bool = True, 
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    MSE loss for ensemble models.

    Parameters:
        y_pred: (batch, ensemble, output_dim)
        y_true: (batch,) or (batch, output_dim)
        training: True -> return per-member loss, False -> average predictions first
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Tensor: scalar if reduction='mean'/'sum', else
                (batch, ensemble) for training or (batch,) for validation
    """
    # Ensure y_true has shape (batch, output_dim)
    if y_true.ndim == 1:
        y_true = y_true.unsqueeze(-1)  # (batch, 1)

    if training:
        # Expand to ensemble dimension
        y_true_exp = y_true.unsqueeze(1)  # (batch, 1, output_dim)
        mse = (y_pred - y_true_exp) ** 2
        mse = mse.mean(dim=-1)  # (batch, ensemble)
    else:
        # Average predictions across ensemble
        avg_pred = y_pred.mean(dim=1)  # (batch, output_dim)
        mse = ((avg_pred - y_true) ** 2).mean(dim=-1)  # (batch,)

    if reduction == 'mean':
        return mse.mean()
    elif reduction == 'sum':
        return mse.sum()
    else:
        return mse

# ------------------------------
# Classification Loss
# ------------------------------
def classification_loss(
    y_pred: torch.Tensor, 
    y_true: torch.Tensor, 
    training: bool = True, 
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Cross-entropy loss for ensemble models.

    Parameters:
        y_pred: (batch, ensemble, num_classes) logits
        y_true: (batch,) integer labels
        training: True -> per-member CE, False -> average probabilities first
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Tensor: scalar if reduction='mean'/'sum', else
                (batch, ensemble) for training or (batch,) for validation
    """
    if training:
        batch, ensemble, num_classes = y_pred.shape
        # Expand y_true to match batch × ensemble
        y_true_exp = repeat(y_true, "b -> (b e)", e=ensemble)
        # Flatten y_pred from (batch, ensemble, num_classes) -> (batch*ensemble, num_classes)
        y_pred_flat = rearrange(y_pred, "b e c -> (b e) c")
        # Compute cross-entropy in one go and reshape back to (batch, ensemble)
        loss_tensor = rearrange(F.cross_entropy(y_pred_flat, y_true_exp, reduction="none"), "(b e) -> b e", b=batch, e=ensemble)
    else:
        probs = F.softmax(y_pred, dim=-1)      # (batch, ensemble, num_classes)
        avg_probs = probs.mean(dim=1)          # (batch, num_classes)
        log_avg_probs = torch.log(avg_probs)   # log-probabilities
        loss_tensor = F.nll_loss(log_avg_probs, y_true,reduction='none')

    if reduction == 'mean':
        return loss_tensor.mean()
    elif reduction == 'sum':
        return loss_tensor.sum()
    else:
        return loss_tensor
