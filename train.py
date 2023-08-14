import os

import hydra
import torch
import wandb
import torchmetrics
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.unet import Unet
from ecg_segmentation_dataset import ECGDataset


def intersection_over_union(input: torch.Tensor, target: torch.Tensor, threshold: float = 0.5):
    # input and target are probabilities of belonging to certain class
    pred = input > threshold
    mask = target > threshold

    intersection = (pred & mask).sum()
    union = (pred | mask).sum()

    # if union is all zeros, it means signal was zeros and model predicted it well, so return 1
    iou = intersection / union if union > 0 else 1

    return iou


def makedir_if_not_exists(dir: str):
    if not os.path.exists(dir):
        os.makedirs(dir)


def preprocess_dataset(dataset_name: str, batch_size: int, num_workers: int):
    train_ds = ECGDataset(dataset_name, split="train")
    val_ds = ECGDataset(dataset_name, split="validation")
    test_ds = ECGDataset(dataset_name, split="test")

    # dataloaders
    train_dataloader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers)
    val_dataloader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers)
    test_dataloader = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers)

    return train_dataloader, val_dataloader, test_dataloader


def save_onnx_model(model: nn.Module, path: str):
    device = next(model.parameters()).device
    # example input for model
    dummy_input = torch.randn((1, 2, 1000), requires_grad=True, device=device)

    # exporting onnx model
    torch.onnx.export(model, dummy_input, path)


def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, save_path: str):
        # saving models
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, save_path)

def step(
        model: nn.Module,
        batch: dict[str, torch.Tensor, torch.Tensor],
        device: torch.device
    ):
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

        return loss, acc, f1, iou


@hydra.main(config_path="configs", config_name="config-default", version_base="1.3.2")
def train(cfg: OmegaConf):
    # create dir if they don't exist
    makedir_if_not_exists(cfg.paths.log_dir)
    makedir_if_not_exists(cfg.paths.save_ckpt_dir)
    makedir_if_not_exists("onnx-models")

    # dataset
    train_dataloader, val_dataloader, test_dataloader = preprocess_dataset(
        dataset_name=cfg.train.dataset_name,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
    )

    # logger
    # logger = WandbLogger(project="ecg-segmentation-unet", name=cfg.logger.run_name, save_dir=cfg.paths.log_dir)
    wandb.init(project="ecg-segmentation-unet", name=cfg.logger.run_name, dir=cfg.paths.log_dir)

    device = torch.device(cfg.train.device)

    # model
    unet = Unet(
        in_channels=cfg.unet.in_channels,
        out_channels=cfg.unet.out_channels,
        dim=cfg.unet.dim,
        dim_mults=cfg.unet.dim_mults,
        kernel_size=cfg.unet.kernel_size,
        resnet_block_groups=cfg.unet.num_resnet_groups,
    ).to(device)

    optimizer = optim.AdamW(unet.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    # ckpt specifies directory and name of the file is name of the experiment in wandb
    save_path = f"{cfg.paths.save_ckpt_dir}/{cfg.logger.run_name}.ckpt"

    train_step_count = 0
    val_step_count = 0

    for epoch in range(cfg.train.num_epochs):
        # train epoch
        train_loop = tqdm(enumerate(train_dataloader))

        for batch_idx, batch in train_loop:
            loss, acc, f1, iou = step(unet, batch, device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loop.set_postfix(loss=loss.item())

            train_step_count += 1

            if (batch_idx + 1) % cfg.logger.log_every_n_steps == 0:
                # log metrics
                wandb.log({"train/loss": loss, "train/accuracy": acc, "train/f1-score": f1, "train/iou": iou}, step=train_step_count)

                # save model and optimizer states
                save_checkpoint(unet, optimizer, save_path=save_path)

        # val epoch
        val_loop = tqdm(enumerate(val_dataloader))

        for batch_idx, batch in val_loop:
            loss, acc, f1, iou = step(unet, batch, device)

            val_loop.set_postfix(loss=loss.item())

            val_step_count += 1

            if (batch_idx + 1) % cfg.logger.log_every_n_steps == 0:
                wandb.log({"val/loss": loss, "val/accuracy": acc, "val/f1-score": f1, "val/iou": iou}, step=val_step_count)



    # testing model


    wandb.finish()

    # save onnx model and log to wandb
    # onnx_save_path = f"onnx-models/model-{cfg.logger.run_name}.onnx"
    # save_onnx_model(unet_wrapper.model, path=onnx_save_path)
    # wandb.save(onnx_save_path)


if __name__ == "__main__":
    wandb.login()

    train()