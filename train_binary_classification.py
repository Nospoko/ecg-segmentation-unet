import torch
import torchmetrics
import torch.nn as nn
import torch.nn.functional as F


def intersection_over_union(input: torch.Tensor, target: torch.Tensor, threshold: float = 0.5):
    # input and target are probabilities of belonging to certain class
    pred = input > threshold
    mask = target > threshold

    intersection = (pred & mask).sum()
    union = (pred | mask).sum()

    # if union is all zeros, it means signal was zeros and model predicted it well, so return 1
    iou = intersection / union if union > 0 else 1

    return iou


def step(model: nn.Module, batch: dict[str, torch.Tensor, torch.Tensor], device: torch.device, split: str = "train") -> dict:
    """
    Step for training binary classification Unet

    Args:
        model (nn.Module): Unet
        batch (dict[str, torch.Tensor, torch.Tensor]): data dict[record_id, signal, mask]
        device (torch.device): cuda or cpu
        split (str, optional): specifies if it's training or validation step. Defaults to "train".

    Returns:
        dict: loss and addictional metrics that will be logged
    """

    # extract signal and mask from batch
    signal = batch["signal"].to(device)
    mask = batch["mask"].to(device)

    mask_logits = model(signal)

    # calculate loss
    loss = F.binary_cross_entropy_with_logits(mask_logits, mask)

    # mask probabilities, used for calculating accuracy and f1 score
    mask_pred = torch.sigmoid(mask_logits)
    # calculate other metrics
    acc = torchmetrics.functional.accuracy(mask_pred, mask, task="binary")
    f1 = torchmetrics.functional.f1_score(mask_pred, mask, task="binary")
    iou = intersection_over_union(mask_pred, mask)

    metrics = {f"{split}/loss": loss, f"{split}/accuracy": acc, f"{split}/f1-score": f1, f"{split}/iou": iou}

    return metrics
