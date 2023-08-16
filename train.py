import os

import hydra
import torch
import wandb
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from models.unet import Unet
from ecg_segmentation_dataset import ECGDataset

from train_binary_classification import step as step_bc
from train_distribution_modelling import step as step_dm


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


def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, cfg: OmegaConf, save_path: str):
    # saving models
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": cfg}, save_path)



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

    # specify task
    if cfg.train.task == "binary-classification":
        step = step_bc
    elif cfg.train.task == "distribution-modelling":
        step = step_dm
    else:
        raise ValueError("No such task")

    # setting up optimizer
    optimizer = optim.AdamW(unet.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    # ckpt specifies directory and name of the file is name of the experiment in wandb
    save_path = f"{cfg.paths.save_ckpt_dir}/{cfg.logger.run_name}.ckpt"

    # step counts for logging to wandb
    train_step_count = 0
    val_step_count = 0

    for epoch in range(cfg.train.num_epochs):
        # train epoch
        train_loop = tqdm(enumerate(train_dataloader))

        for batch_idx, batch in train_loop:
            # metrics returns loss and additional metrics if specified in step function
            metrics = step(unet, batch, device, split="train")

            loss = metrics["train/loss"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loop.set_postfix(loss=loss.item())

            train_step_count += 1

            if (batch_idx + 1) % cfg.logger.log_every_n_steps == 0:
                # log metrics
                wandb.log(metrics, step=train_step_count)

                # save model and optimizer states
                save_checkpoint(unet, optimizer, cfg, save_path=save_path)

        # val epoch
        val_loop = tqdm(enumerate(val_dataloader))

        for batch_idx, batch in val_loop:
            metrics = step(unet, batch, device, split="val")

            loss = metrics["val/loss"]

            val_loop.set_postfix(loss=loss.item())

            val_step_count += 1

            if (batch_idx + 1) % cfg.logger.log_every_n_steps == 0:
                wandb.log(metrics, step=val_step_count)


    wandb.finish()


if __name__ == "__main__":
    wandb.login()

    train()
