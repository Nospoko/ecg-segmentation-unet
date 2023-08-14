import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
import torchmetrics

from models.unet import Unet
from ecg_segmentation_dataset import ECGDataset
from train import preprocess_dataset, intersection_over_union

if __name__ == "__main__":
    # initializing model
    ckpt = torch.load("checkpoints/run-2023-08-14-09-24.ckpt")
    model = Unet(in_channels=2, out_channels=1, dim=32, dim_mults=(1, 2, 4), kernel_size=7, resnet_block_groups=4)
    model.load_state_dict(ckpt["model"])

    acc = torchmetrics.Accuracy("binary")
    f1 = torchmetrics.F1Score("binary")

    train_dataloader, val_dataloader, test_dataloader = preprocess_dataset("roszcz/ecg-segmentation-ltafdb", 128, 1)

    # initialing random input
    record = next(iter(test_dataloader))
    signal = record["signal"]
    mask = record["mask"]

    # print shape
    pred_mask_logits = model(signal)
    pred_mask = torch.sigmoid(pred_mask_logits) > 0.5

    indices = [idx for idx in range(len(mask)) if mask[idx].sum() > 0]
    # idx = indices[0]

    m = mask.detach()
    pred_m = pred_mask.detach()

    # plot
    fig, axes = plt.subplots(4, 2, figsize=(15, 7))

    for i, ax in enumerate(axes):
        idx = indices[i]

        a = acc(pred_m[idx], m[idx])
        f = f1(pred_m[idx], m[idx])
        iou = intersection_over_union(pred_m[idx], m[idx])
        print(f"Sample {i} acc: {a}, f1: {f}, iou: {iou}")


        sns.lineplot(mask[idx, 0, :], ax=axes[i, 0])
        sns.lineplot(pred_m[idx, 0, :], alpha=0.8, ax=axes[i, 0])
        axes[i, 0].set_title("Mask")

        sns.lineplot(signal[idx, 0], ax=axes[i, 1])
        sns.lineplot(signal[idx, 1], ax=axes[i, 1])
        axes[i, 1].set_title("Signal")




    plt.tight_layout()
    plt.show()