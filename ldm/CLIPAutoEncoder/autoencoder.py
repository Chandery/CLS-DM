import torch
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

# import pytorch_lightning as pl
import lightning as pl
import torch.nn.functional as F
import torch.nn as nn
from contextlib import contextmanager
import hydra
import numpy as np
import SimpleITK as sitk
import os
import torchvision
from monai.transforms import SaveImage

from autoencoderkl.autoencoder import AutoencoderKL
from Medicalnet.VIT3D import VisionTransformer


class VQModelInterface:
    def __init__(self) -> None:
        pass


class CLIPAE(AutoencoderKL):
    def __init__(
        self,
        config,
        freeze_first_stage=False,
    ):
        super().__init__(save_path=config.hydra_path, config=config, **config["model"])

        # ! ---------------init cond_stage_model----------------
        model = VisionTransformer(img_size=config.cond_size, patch_size=16, in_chans=1, embed_dim=8*16*16*16, num_heads=16, drop_rate=0.1, attn_drop_rate=0.1, drop_path_rate=0.1)
        
        self.cond_stage_model = model
        # ! ---------------init cond_stage_model----------------
    
    def condition_vit_encode(self, cond):
        """
        using vit backbone to encode conditioning x-ray imgs.
        backbone checkpoint from https://github.com/duyhominhnguyen/LVM-Med
        input: (1,1,256,256)
        output: (1,4,16,16,16) match the latent code z.
        """
        # Note output shape is (1,8,16,16,16) not (1,4,16,16,16)
        # Note channel is not controlled by config, it's fixed to 8
        # ? fix: (1,1,256,256) -> (1,8,16,16,16)
        
        B, C, H, W = cond.shape
        cond = cond.repeat(1, 1, 1, 1, H) 
        cond = self.cond_stage_model(cond)
        return cond

    def training_step(self, batch, batch_idx):
        opt_ae, opt_disc, opt_cond= self.optimizers()

        # ? ----------------get inputs----------------

        inputs = batch["image"]
        x = torch.as_tensor(batch["image"])
        if 1 in self.cond_nums:
            # cond1 = batch["cond1"].as_tensor()
            cond1 = torch.as_tensor(batch["cond1"])
        else:
            cond1 = None
        if 2 in self.cond_nums:
            # cond2 = batch["cond2"].as_tensor()
            cond2 = torch.as_tensor(batch["cond2"])
        else:
            cond2 = None
        if 3 in self.cond_nums:
            # cond3 = batch["cond3"].as_tensor()
            cond3 = torch.as_tensor(batch["cond3"])
        else:
            cond3 = None
        # print(inputs.shape)


        # ? ----------------training----------------
        reconstructions, posterior = self(inputs)

        # train encoder+decoder+logvar
        aeloss, log_dict_ae = self.loss(
            inputs,
            reconstructions,
            posterior,
            0,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="train",
        )
        opt_ae.zero_grad()
        self.manual_backward(aeloss)
        opt_ae.step()

        self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=self.sync_dist)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)

        # train the discriminator
        discloss, log_dict_disc = self.loss(
            inputs,
            reconstructions,
            posterior,
            1,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="train",
        )
        opt_disc.zero_grad()
        self.manual_backward(discloss)
        opt_disc.step()

        self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=self.sync_dist)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False, sync_dist=self.sync_dist)

    def validation_step(self, batch, batch_idx):
        if self.current_epoch % 10 == 0:
            inputs = batch["image"]
            # print(inputs.shape)
            reconstructions, posterior = self(inputs)

            # reconstructions = torch.clamp(reconstructions, min=-1, max=1)
            # reconstructions = (reconstructions + 1) * 127.5

            # inputs = torch.clamp(inputs, min=-1, max=1)
            # inputs = (inputs + 1) * 127.5

            rec_loss = F.mse_loss(reconstructions, inputs)

            # reconstructions = reconstructions.squeeze(0).permute(1, 0, 2, 3)
            # reconstructions = reconstructions.type(torch.uint8)
            # grid = torchvision.utils.make_grid(reconstructions)
            # self.logger.experiment.add_image("val_images", grid, self.global_step)

            # inputs = inputs.type(torch.uint8)
            # inputs = inputs.squeeze(0).permute(1, 0, 2, 3)
            # inputs = inputs.type(torch.uint8)
            # grid = torchvision.utils.make_grid(inputs)
            # self.logger.experiment.add_image("val_inputs", grid, self.global_step)

            self.log("val/rec_loss", rec_loss, sync_dist=self.sync_dist)

    def img_saver(self, img, post_fix, i_type=".nii", meta_data=None, filename=None, **kwargs):
        """
        save img to self.root_path with post_fix

        Args:
            img (torch.Tensor): [description]
            post_fix (str): [description]
            type (str, optional): [description]. Defaults to "nii".
            meta_data ([type], optional): [description]. Defaults to None.
        """
        if hasattr(img, "meta"):
            meta_data = img.meta
        else:
            print("img dosen't has meta attribution use `None` as meta_dat")

        assert i_type in [".nii", ".nii.gz", ".jpg"], "Only .nii or .jpg suffix file supported now"
        # assert post_fix in ["origin_x", "ae_rec", "xray1", "xray2", "rec"], "unsupported post_fix"

        img = img.squeeze(0)
        print(f"max value :{torch.max(img)}")
        print(f"min value :{torch.min(img)}")
    
        # if post_fix == "ae_rec":
        #     MAX = torch.max(img)
        #     MIN = torch.min(img)
        #     img = 2*(img-MAX)/(MAX-MIN)-1
        # else:
        img = torch.clamp(img, min=-1, max=1)
        # img = (img + 1)/2  # scale to 0-1
        # img = img * (self.config.CT_MIN_MAX[1]-self.config.CT_MIN_MAX[0]) + self.config.CT_MIN_MAX[0]
        img = (img + 1) * 127.5
        writer = "NibabelWriter" if "nii" in i_type else "PILWriter"
        out_ext = ".nii.gz" if "nii" in i_type else ".jpg"

        saver = SaveImage(
            output_dir=self.root_path,
            output_ext=out_ext,
            output_postfix=post_fix,
            separate_folder=False,
            output_dtype=np.uint8,
            resample=False,
            squeeze_end_dims=True,
            writer=writer,
            **kwargs,
        )
        # saver(img, meta_data=meta_data)
        saver(img, filename=filename)

    def test_step(self, batch, batch_idx):
        inputs = batch["image"]
        reconstructions, posterior = self(inputs)
        # reconstructions = to_image(reconstructions)
        self.img_saver(inputs, post_fix="origin_x", filename=str(batch_idx)+"_origin_x")
        self.img_saver(reconstructions, post_fix="ae_rec", filename=str(batch_idx)+"_ae_rec")

        # image = sitk.GetImageFromArray(reconstructions)
        # sitk.WriteImage(image, os.path.join(self.save_path, f"reconstructions_{batch_idx}.mhd"))

        # inputs = to_image(inputs)
        # image = sitk.GetImageFromArray(inputs)
        # sitk.WriteImage(image, os.path.join(self.save_path, f"origin_{batch_idx}.mhd"))
        # save_path = os.path.join(self.save_path, "val_reconstruction")
        # os.makedirs(save_path, exist_ok=True)
        # image = sitk.GetImageFromArray(reconstructions)
        # sitk.WriteImage(image, os.path.join(save_path, f"{self.global_step}.mhd"))

        # self.log("val/rec_loss", log_dict_ae["val/rec_loss"], sync_dist=self.sync_dist)
        # self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False, sync_dist=self.sync_dist)
        # self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False, sync_dist=self.sync_dist)
        # return self.log_dic

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.quant_conv.parameters())
            + list(self.post_quant_conv.parameters()),
            lr=lr,
            betas=(0.5, 0.9),
        )
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(), lr=lr, betas=(0.5, 0.9))
        opt_cond = torch.optim.Adam(self.cond_stage_model.parameters(), lr=lr, betas=(0.5, 0.9))
        return opt_ae, opt_disc, opt_cond

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def to_image(x):
        x = torch.clamp(x, min=-1, max=1)
        x = (x + 1) * 127.5
        # x = x.squeeze(0).permute(1, 0, 2, 3)
        x = x.type(torch.uint8)
        x = x.cpu().numpy()
        return x


@hydra.main(version_base=None, config_path="../../conf", config_name="/config/autoencoder.yaml")
def main(config):
    # config = config["config"]
    # model_config = config["model"]
    # ddconfig = config["model"]["params"]["ddconfig"]
    # lossconfig = config["model"]["params"]["lossconfig"]
    # print(model_config.get("params", dict()))
    # model = AutoencoderKL(model_config)
    input = torch.randn((1, 1, 256, 256))
    # output, _ = model(input)
    # print(output.shape)
    model = VisionTransformer(img_size=config.cond_size, patch_size=16, in_chans=1, embed_dim=8*16*16*16, num_heads=16, drop_rate=0.1, attn_drop_rate=0.1, drop_path_rate=0.1)

    y = model(input)
    print(y.shape)


if __name__ == "__main__":
    main()
