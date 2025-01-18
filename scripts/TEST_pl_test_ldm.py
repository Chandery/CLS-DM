import hydra
import torch
from torch.utils.data import DataLoader
import sys
import os
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from dataset.monai_nii_dataset import prepare_dataset

# from ldm.autoencoderkl.autoencoder import AutoencoderKL
from ldm.ddpm import LatentDiffusion

torch.set_float32_matmul_precision("high")


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def train(config):
    config = config["config"]
    # * dataset and dataloader
    test_dl = prepare_dataset(
        cond_path=config.cond_path,
        data_path=config.data_path,
        resize_size=config.resize_size,
        img_resize_size=config.img_resize_size,
        split="test",
    )
    # * model
    model = LatentDiffusion(root_path=config.hydra_path, **config["latent_diffusion"])
    # model.init_from_ckpt(
    #     "/disk/cc/Xray-Diffsuion/logs/ldm/pl_train_ldm-2024-09-05/16-53-31-zhougu/pl_train_autoencoder-epoch1740-val_rec_loss0.00.ckpt"
    # )
    model.init_from_ckpt("/disk/cc/Xray-Diffsuion/logs/ldm/pl_train_ldm-2024-09-16/23-32-56/latest.ckpt"
    )

    # * trainer fit
    trainer = pl.Trainer(**config["trainer"], default_root_dir=config.hydra_path)
    trainer.test(model=model, dataloaders=test_dl)


if __name__ == "__main__":
    train()
