import hydra
import os
import torch
from torch.utils.data import DataLoader
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from lightning.pytorch.loggers import NeptuneLogger, CometLogger

# from dataset.med_3Ddataset import ImageDataset
from dataset.MhdDataset import MhdDataset
import lightning as pl
from ldm.autoencoderkl.autoencoder import AutoencoderKL
from lightning.pytorch.callbacks import ModelCheckpoint
from dataset.monai_nii_dataset import prepare_dataset
from dataset.monai_nii_dataset1 import AlignDataSet
from lightning.pytorch.strategies import DDPStrategy
from ldm.CLIPAutoEncoder.autoencoder import CLIPAE

torch.set_float32_matmul_precision("high")  


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def train(config):
    config = config["config"]
    checkpoint_callback = ModelCheckpoint(
        monitor=config["model"].monitor,
        dirpath=config.hydra_path,
        filename="pl_train_autoencoder-epoch{epoch:02d}-val_rec_loss{val/rec_loss:.2f}",
        save_top_k=1,
        mode="min",
        auto_insert_metric_name=False,
    )
    checkpoint_callback_latest = ModelCheckpoint(
        dirpath=config.hydra_path,
        filename="latest",
    )
    train_ds = AlignDataSet(config,split = "train")
    train_dl = DataLoader(
        dataset=train_ds,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=config.num_workers,
        batch_size=config.batch_size,
    )
    val_ds = AlignDataSet(config, split = "val")
    val_dl = DataLoader(
        dataset=val_ds,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
        num_workers=config.num_workers,
        batch_size=config.batch_size,
    )
    # n = next(iter(train_dl))
    # print(n["image"].shape) # ? torch.Size([2, 1, 128, 128, 128])
    # print(n["cond1"].shape) # ? torch.Size([1, 1, 256, 256])
    # print(n["cond2"].shape) # ? torch.Size([1, 1, 256, 256])
    # return

    # * model
    model = CLIPAE(config=config)

    # * trainer fit
    trainer = pl.Trainer(
        **config["trainer"],
        callbacks=[checkpoint_callback, checkpoint_callback_latest],
        default_root_dir=config.hydra_path,
    )
    trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=val_dl)


if __name__ == "__main__":
    train()
