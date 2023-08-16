import torch
import torch.nn as nn
import torch.nn.functional as F


def step(model: nn.Module, batch: dict[str, torch.Tensor, torch.Tensor], device: torch.device, split: str = "train") -> dict:
    """
    Step for training distribution modelling Unet

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

    prediction = model(signal)

    # calculate loss
    loss = F.mse_loss(prediction, mask)

    metrics = {
        f"{split}/loss": loss
    }

    return metrics