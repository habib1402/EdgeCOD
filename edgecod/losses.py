"""Training loss (Section 3.5): BCE + lambda * IoU, and the optional boundary Dice term of the loss study."""
import torch
import torch.nn.functional as F


def bce_iou(logits, mask, iou_weight=0.5, eps=1e-7):
    """BCE + lambda * soft IoU, computed from logits for numerical stability (lambda = 0.5 in the paper)."""
    bce = F.binary_cross_entropy_with_logits(logits, mask)
    p = torch.sigmoid(logits)
    inter = (p * mask).sum((2, 3))
    union = (p + mask - p * mask).sum((2, 3))
    iou = 1 - (inter + eps) / (union + eps)
    return bce + iou_weight * iou.mean()


def edge(x, k=3):
    """Soft boundary map: morphological gradient (dilation minus erosion, k x k) of a [0, 1] map."""
    return F.max_pool2d(x, k, 1, k // 2) + F.max_pool2d(-x, k, 1, k // 2)


def boundary_dice(logits, mask, eps=1.0):
    """Dice loss between the predicted and ground-truth boundary maps (loss study, Table 15; not used by default)."""
    pe, ge = edge(torch.sigmoid(logits)), edge(mask)
    inter = (pe * ge).sum((2, 3))
    return (1 - (2 * inter + eps) / (pe.sum((2, 3)) + ge.sum((2, 3)) + eps)).mean()


def total_loss(logits, mask, cfg):
    logits = logits.float()
    loss = bce_iou(logits, mask, cfg.get('iou_weight', 0.5))
    if cfg.get('boundary_weight'):
        loss = loss + cfg['boundary_weight'] * boundary_dice(logits, mask)
    return loss
