import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torchmetrics
import wandb
import hydra
import os

from torch.utils.data import DataLoader, random_split
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint

from omegaconf import OmegaConf

from models.unet import Unet
from ecg_segmentation_dataset import ECGDataset


def intersection_over_union(input: torch.Tensor, target: torch.Tensor, threshold: float = 0.5):
    # input and target are probabilities of belonging to certain class
    pred = input > threshold
    mask = target > threshold

    iou = (pred & mask).sum() / (pred | mask).sum()

    return iou

class UnetTrainingWrapper(pl.LightningModule):
    def __init__(self, model: Unet, lr: float, weight_decay: float):
        super().__init__()

        self.model = model

        self.save_hyperparameters(ignore=[model])

        # metrics
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.accuracy = torchmetrics.Accuracy(task="binary")
        self.f1_score = torchmetrics.F1Score("binary")


    def forward(self, x: torch.Tensor):
        return self.model(x)
    
    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = optim.AdamW(self.model.parameters(), lr=self.hparams["lr"], weight_decay=self.hparams["weight_decay"])

        return optimizer
    
    def _step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        # extract signal and mask from batch
        signal, mask = batch

        mask_logits = self.model(signal)

        # calculate loss
        loss = self.loss_fn(mask_logits, mask)

        # mask probabilities, used for calculating accuracy and f1 score
        mask_pred = torch.sigmoid(mask_logits)
        # calculate other metrics
        acc = self.accuracy(mask_pred, mask)
        f1 = self.f1_score(mask_pred, mask)
        iou = intersection_over_union(mask_pred, mask)

        return loss, acc, f1, iou
    
    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        loss, acc, f1, iou = self._step(batch, batch_idx)

        self.log_dict(
            {
                "train/loss": loss,
                "train/accuracy": acc,
                "train/f1-score": f1,
                "train/iou": iou
            },
            on_epoch=True
        )

        return loss
    
    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        loss, acc, f1, iou = self._step(batch, batch_idx)

        self.log_dict(
            {
                "val/loss": loss,
                "val/accuracy": acc,
                "val/f1-score": f1,
                "train/iou": iou
            },
            on_epoch=True
        )

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        loss, acc, f1, iou = self._step(batch, batch_idx)

        self.log_dict(
            {
                "test/loss": loss,
                "test/accuracy": acc,
                "test/f1-score": f1,
                "train/iou": iou
            },
            on_epoch=True
        )



def makedir_if_not_exists(dir: str):
    if not os.path.exists(dir):
        os.makedirs(dir)


def preprocess_dataset(dataset_name: str, batch_size: int, num_workers: int, seed: int = 0):
    dataset = ECGDataset(dataset_name)

    # generator used to determine which records belong to which split
    generator = torch.Generator().manual_seed(seed)

    # spliting data based on seed
    train_ds, val_ds, test_ds = random_split(dataset, [0.8, 0.1, 0.1], generator=generator)

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


@hydra.main(config_path="configs", config_name="config-default", version_base="1.3.2")
def train(cfg: OmegaConf):
    
    # create dir if they don't exist
    makedir_if_not_exists(cfg.paths.log_dir)
    makedir_if_not_exists(cfg.paths.save_ckpt_dir)
    makedir_if_not_exists("onnx-models")

    # dataset
    train_dataloader, val_dataloader, test_dataloader = preprocess_dataset(
        dataset_name=cfg.hyperparameters.dataset_name,
        batch_size=cfg.hyperparameters.batch_size,
        num_workers=cfg.hyperparameters.num_workers,
        seed=cfg.hyperparameters.dataset_split_seed,
    )

    # logger
    logger = WandbLogger(project="ecg-segmentation-unet", name=cfg.logger.run_name, save_dir=cfg.paths.log_dir)

    # model
    unet = Unet(
        in_channels=cfg.unet.in_channels,
        out_channels=cfg.unet.out_channels,
        dim=cfg.unet.dim,
        dim_mults=cfg.unet.dim_mults,
        kernel_size=cfg.unet.kernel_size,
        resnet_block_groups=cfg.unet.num_resnet_groups,
    )

    # lightning training wrapper
    unet_wrapper = UnetTrainingWrapper(
        unet, 
        lr=cfg.hyperparameters.lr, 
        weight_decay=cfg.hyperparameters.weight_decay
    )

    # load checkpoint if specified
    if cfg.paths.load_ckpt_path is not None:
        unet_wrapper.load_from_checkpoint(checkpoint_path=cfg.paths.load_ckpt_path)

    
    # callbacks
    callbacks = [
        TQDMProgressBar(), 
        ModelCheckpoint(dirpath=cfg.paths.save_ckpt_dir, monitor="val/loss")
    ]

    # initializing trainer with specified hyperparameters
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        max_epochs=cfg.hyperparameters.num_epochs,
        accelerator=cfg.hyperparameters.accelerator,
        precision=cfg.hyperparameters.precision,
        overfit_batches=cfg.hyperparameters.overfit_batches,
        log_every_n_steps=cfg.logger.log_every_n_steps,
    )

    # run training
    trainer.fit(
        unet_wrapper,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
    )

    # test model
    trainer.test(unet_wrapper, test_dataloader)

    # save onnx model and log to wandb
    onnx_save_path = f"onnx-models/model-{cfg.logger.run_name}.onnx"
    save_onnx_model(unet_wrapper.model, path=onnx_save_path)
    wandb.save(onnx_save_path)


if __name__ == "__main__":
    wandb.login()

    train()

    wandb.finish()