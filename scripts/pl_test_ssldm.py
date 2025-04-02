import hydra
import torch
from torch.utils.data import DataLoader
import sys
import os
import lightning as pl
# from lightning.pytorch.callbacks import ModelCheckpoint


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
# from dataset.monai_nii_dataset import prepare_dataset

# from ldm.autoencoderkl.autoencoder import AutoencoderKL
from ldm.ddpm import LatentDiffusion
from dataset.monai_nii_dataset1 import AlignDataSet
# from lightning.pytorch.strategies import DDPStrategy

torch.set_float32_matmul_precision("high")


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def train(config):
    config = config["config"]
    # * dataset and dataloader
    # test_dl = prepare_dataset(
    #     cond_path=config.cond_path,
    #     data_path=config.data_path,
    #     resize_size=config.resize_size,
    #     img_resize_size=config.img_resize_size,
    #     split="test",
    #     cond_nums=config.latent_diffusion.cond_nums,
    # )
    test_ds = AlignDataSet(config, split = "val")
    test_dl = DataLoader(
        dataset=test_ds,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
        num_workers=config.num_workers,
        batch_size=1,
    )

    # * model
    model = LatentDiffusion(config=config, root_path=config.hydra_path, **config["latent_diffusion"])
    # model.init_from_ckpt(
    #     "/disk/cc/Xray-Diffsuion/logs/ldm/pl_train_ldm-2024-11-06/10-55-23-zhougu/latest.ckpt"
    # )
    ckpt = "/home/cdy/SharedSpaceLDM/logs/ssldm/pl_train_ssldm-2025-03-31/14-05-56/pl_train_ssldm-epoch719-val_ssim0.642.ckpt"
    model.init_from_ckpt(ckpt)
    # model.init_from_ckpt(
    #     "/disk/cc/Xray-Diffsuion/logs/ldm/pl_train_ldm-2024-11-04-pengu/02-21-15/pl_train_autoencoder-epoch1110-val_rec_loss0.00.ckpt"
    # )

    # * trainer fit
    trainer = pl.Trainer(**config["trainer"], default_root_dir=config.hydra_path)
    trainer.test(model=model, dataloaders=test_dl)


if __name__ == "__main__":
    train()
