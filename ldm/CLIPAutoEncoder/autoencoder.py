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
# from Medicalnet.VIT3D import VisionTransformer
from Medicalnet.unet3d import UNet3D
import torch.utils.checkpoint as checkpoint


class VQModelInterface:
    def __init__(self) -> None:
        pass


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class SpatialAligner(nn.Module):
    def __init__(self, in_channels=16, out_channels=4):
        super().__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.attention = nn.MultiheadAttention(embed_dim=out_channels, num_heads=4, batch_first=False) # ? batch_first=False，表示输入的格式为(seq, batch, feature)   
    
    def forward(self, x):
        x = self.conv3d(x)  # 维度对齐
        x = x.flatten(2).permute(2,0,1)  # [N,C,D,H,W] -> [D*H*W,N,C] # ? 为了符合MultiheadAttention的输入格式
        x, _ = self.attention(x, x, x)
        return x.permute(1,2,0).view(-1,4,16,16,16) 
    # def forward(self, x):
        # return checkpoint.checkpoint(self._forward, x)

class CLIPAE(pl.LightningModule):
    def __init__(
        self,
        save_path: str,
        config,
    ):
        super().__init__()
        self.automatic_optimization = False

        self.cond_nums = config.cond_nums
        self.cond_num = (1 in self.cond_nums) + (2 in self.cond_nums) + (3 in self.cond_nums)

        self.cond1_order = list(config.cond1_order)
        self.cond2_order = list(config.cond2_order)
        self.cond3_order = list(config.cond3_order)

        self.condloss_ratio = config.condloss_ratio

        self.cond_config = config.cond_model_config
        
        self.cond_type = config.cond_type

        self.in_c =1 if self.cond_type=='add' else self.cond_num

        self.loss_ratio = config.loss_ratio
        self.xray_size = config.fine_size_cond

        self.learning_rate = config.model.base_learning_rate
        self.root_path = save_path
        self.sync_dist = config.model.sync_dist

        # ! ---------------init cond_stage_model----------------
        model = UNet3D(in_channels=self.in_c, out_channels=1, init_features=64, use_checkpoint=False) # ? use_checkpoint=True，可以减少显存占用，但是速度会变慢
        self.cond_stage_model = model
        self.proj_head = SpatialAligner(in_channels=16, out_channels=4)
        self.up_dim_head = nn.Sequential(
            # 首先在深度方向上扩展到self.xray_size
            nn.ConvTranspose3d(in_channels=1, out_channels=8, kernel_size=(self.xray_size//4,1,1), stride=(self.xray_size//4,1,1), padding=(0,0,0)),
            nn.BatchNorm3d(num_features=8),
            nn.ReLU(inplace=False),
            nn.ConvTranspose3d(in_channels=8, out_channels=16, kernel_size=(2,1,1), stride=(2,1,1), padding=(0,0,0)),
            nn.BatchNorm3d(num_features=16),
            nn.ReLU(inplace=False),
            nn.ConvTranspose3d(in_channels=16, out_channels=16, kernel_size=(2,1,1), stride=(2,1,1), padding=(0,0,0)),
            nn.BatchNorm3d(num_features=16),
            nn.ReLU(inplace=False),
            # 然后在高度和宽度方向上使用1x1卷积学习特征
            nn.Conv3d(in_channels=16, out_channels=1, kernel_size=1),
        )
        # ! ---------------init cond_stage_model----------------

        self.init_ae_model(save_path, config)
    
    def init_ae_model(self, save_path, config):
        model = AutoencoderKL(save_path=save_path, config=config, **config["model"])
        model.init_from_ckpt(config.ae_ckpt)
        model.eval()
        model.train = disabled_train
        for param in model.parameters():
            param.requires_grad = False
        self.ae_model = model
        # self.encoder = model.encoder
        # self.decoder = model.decoder
        # self.quant_conv = model.quant_conv
        # self.post_quant_conv = model.post_quant_conv
        # self.encode = model.encode
        # self.decode = model.decode
        print("init ae model success")
    
    @property # ? 使用这个property装饰器，可以使得encoder等属性不用重复存储，直接返回ae_model的属性
    def encoder(self):
        return self.ae_model.encoder
    @property
    def decoder(self):
        return self.ae_model.decoder
    @property
    def encode(self):
        return self.ae_model.encode
    @property
    def decode(self):
        return self.ae_model.decode
    @property
    def quant_conv(self):
        return self.ae_model.quant_conv
    @property
    def post_quant_conv(self):
        return self.ae_model.post_quant_conv
    
    def conditional_encode(self, cond):
        """
        using UNet3D to encode conditioning x-ray imgs.
        input: (1,1,128,128)
        output: 2 tensors,
            c_proj: (1,4,16,16,16)
            cond: (1,16,16,16,16)
            c_proj is the projected condition, which is used to contrast with the latent code z.
            cond is the condition, which is the real condition
        """
        cond, cond_rec = self.cond_stage_model(cond) # ? (1,1,128,128) -> (1,16,16,16,16)
        return cond, cond_rec  # ? (1,4,16,16,16) match the latent code z.
    def cond_up_dim(self, cond):
        """
        Repeat the condition imgs to 5D tensor.
        """
        assert len(cond.shape) == 4, "condition imgs should be 4D tensor, but got {}".format(cond.shape)
        cond = self.up_dim_head(cond.unsqueeze(2))
        return cond
    def contrastive_loss(self, z, c, proj_head, temperature=0.1):
        c = proj_head(c)
        z = F.normalize(z, p=2, dim=1)
        c = F.normalize(c, p=2, dim=1)

        recon_loss = F.mse_loss(c, z)

        c_flat = c.reshape(c.size(0), -1)
        z_flat = z.reshape(z.size(0), -1)

        logits = torch.mm(z_flat, c_flat.T) / temperature
        labels = torch.arange(z.size(0)).to(z.device)  # ? contrastive_loss 函数中使用了 checkpoint.checkpoint。当使用梯度检查点时，有时候会影响 Lightning 的自动设备管理
        return F.cross_entropy(logits, labels) + 0.5*recon_loss
    
    def forward(self, input, sample_posterior=True):
        posterior = self.ae_model.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.ae_model.decode(z)
        return dec, posterior, z
    # ? 使用checkpoint.checkpoint，可以减少显存占用，但是速度会变慢
    # def forward(self, input, sample_posterior=True):
        # return checkpoint.checkpoint(self._forward, input, sample_posterior)
    
    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())  
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")
    def get_cond(self, batch, type='add'):
        cond = []
        if 1 in self.cond_nums:
            cond1 = torch.as_tensor(batch["cond1"])
            cond1 = self.cond_up_dim(cond1).permute(self.cond1_order)
            cond.append(cond1)
        else:
            cond1 = None
        if 2 in self.cond_nums:
            cond2 = torch.as_tensor(batch["cond2"])
            cond2 = self.cond_up_dim(cond2).permute(self.cond2_order)
            cond.append(cond2)
        else:
            cond2 = None
        if 3 in self.cond_nums:
            cond3 = torch.as_tensor(batch["cond3"])
            cond3 = self.cond_up_dim(cond3).permute(self.cond3_order)
            cond.append(cond3)
        else:
            cond3 = None

        cond_cat = torch.cat(cond, dim=1) if self.cond_num != 0 else None # ? (1, 2, 128, 128, 128)
        cond_sum = cond1
        if self.cond_num == 2:
            cond_sum = cond1 + cond2
        elif self.cond_num == 3:
            cond_sum = cond1 + cond2 + cond3
        
        cond_avg = (cond_sum)/self.cond_num # ? (1, 1, 128, 128, 128)

        assert type=="add" or type=="cat", "cond type should be add or cat"

        # print(type)
        
        cond_ret = cond_avg if type=="add" else cond_cat

        return cond_ret

    def training_step(self, batch, batch_idx):
        opt_cond= self.optimizers()

        # ? ----------------get inputs----------------

        # ? get image
        inputs = batch["image"] # ? (1, 1, 128, 128, 128)

        # ? get condition imgs & repeat & permute & concat

        cond_cat = self.get_cond(batch, type=self.cond_type) # ? cat: (1, 2, 128, 128, 128), add: (1, 1, 128, 128, 128)


        # ? ----------------training----------------
        reconstructions, posterior, z = self(inputs)

        cond_latent, cond_rec = self.conditional_encode(cond_cat)

        condloss= self.contrastive_loss(z, cond_latent, self.proj_head)
        rec_loss = F.mse_loss(cond_rec, inputs)

        cliploss = condloss + self.loss_ratio*rec_loss

        opt_cond.zero_grad()
        self.manual_backward(cliploss)
        opt_cond.step()

        self.log("condloss", condloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=self.sync_dist)
        self.log("rec_loss", rec_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=self.sync_dist)
        self.log("cliploss", cliploss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=self.sync_dist)

        # ! check model params
        has_nan = self.check_model_params_detailed(self.cond_stage_model, "cond_stage_model")
        if has_nan:
            # 1. 记录当前的损失值
            nan_info = {
                "nan_detected/condloss": condloss.item() if not torch.isnan(condloss).all() else -1,
                "nan_detected/rec_loss": rec_loss.item() if not torch.isnan(rec_loss).all() else -1,
                "nan_detected/cliploss": cliploss.item() if not torch.isnan(cliploss).all() else -1,
                
                # 2. 记录当前的学习率
                "nan_detected/learning_rate": self.learning_rate,
                
                # 3. 记录关键张量的统计信息
                "nan_detected/z_stats/mean": z.mean().item() if not torch.isnan(z).all() else -1,
                "nan_detected/z_stats/std": z.std().item() if not torch.isnan(z).all() else -1,
                "nan_detected/z_stats/max": z.max().item() if not torch.isnan(z).all() else -1,
                "nan_detected/z_stats/min": z.min().item() if not torch.isnan(z).all() else -1,
                
                "nan_detected/cond_latent_stats/mean": cond_latent.mean().item() if not torch.isnan(cond_latent).all() else -1,
                "nan_detected/cond_latent_stats/std": cond_latent.std().item() if not torch.isnan(cond_latent).all() else -1,
                
                # 4. 记录训练状态
                "nan_detected/epoch": self.current_epoch,
                "nan_detected/global_step": self.global_step,
                "nan_detected/batch_idx": batch_idx,
                
                # 5. 记录梯度信息
                "nan_detected/grad_norm": torch.nn.utils.clip_grad_norm_(self.parameters(), float('inf')).item(),
            }
            
            # 记录到日志
            self.log_dict(nan_info, prog_bar=False, logger=True, sync_dist=self.sync_dist)
            
            # 可选：保存当前批次的输入数据，以便复现
            self.save_debug_info(batch, "nan_detected_batch")
            
            # 可选：提前停止训练
            raise RuntimeError("NaN detected in model parameters!")
        
    def save_debug_info(self, batch, prefix):
        """保存调试信息到文件"""
        debug_dir = os.path.join(self.root_path, "debug_info")
        os.makedirs(debug_dir, exist_ok=True)
        
        # 保存输入数据
        for key, value in batch.items():
            if torch.is_tensor(value):
                save_path = os.path.join(debug_dir, f"{prefix}_{key}.pt")
                torch.save(value.cpu(), save_path)
        
        # 保存当前模型状态
        model_path = os.path.join(debug_dir, f"{prefix}_model.pt")
        torch.save({
            'model_state_dict': self.state_dict(),
            'epoch': self.current_epoch,
            'global_step': self.global_step,
        }, model_path)

    def validation_step(self, batch, batch_idx):
        if self.current_epoch % 10 == 0:
            inputs = batch["image"]
    
            cond_cat = self.get_cond(batch, self.cond_type)
            # print(inputs.shape)
            reconstructions, _, z= self(inputs)
        
            cond_latent, cond_rec = self.conditional_encode(cond_cat)
            condloss= self.contrastive_loss(z, cond_latent, self.proj_head)
            rec_loss = F.mse_loss(cond_rec, inputs)
            cliploss = condloss + self.loss_ratio*rec_loss

            cond_proj = self.proj_head(cond_latent)

            cond_base_rec = self.ae_model.decode(cond_proj) 
            cond_base_rec_loss = F.mse_loss(cond_base_rec, inputs)

            self.log("val/cliploss", cliploss, sync_dist=self.sync_dist)
            self.log("val/condloss", condloss, sync_dist=self.sync_dist)
            self.log("val/rec_loss", rec_loss, sync_dist=self.sync_dist)
            self.log("val/cond_base_rec_loss", cond_base_rec_loss, sync_dist=self.sync_dist)

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
    def check_tensor(self, tensor, name="tensor"):
        print(f"\nChecking {name}:")
        print(f"Shape: {tensor.shape}")
        print(f"Has NaN: {torch.isnan(tensor).any()}")
        print(f"Min: {tensor.min()}")
        print(f"Max: {tensor.max()}")
        print(f"Mean: {tensor.mean()}")
        print(f"Std: {tensor.std()}")
    def check_model_params_detailed(self, model, model_name="model"):
        log_dict = {}
        has_nan = False
        
        for name, param in model.named_parameters():
            # 检查是否有 NaN
            param_has_nan = torch.isnan(param).any().item()
            has_nan = has_nan or param_has_nan
            # 创建该参数的统计信息字典
            param_stats = {
                # f"{model_name}/{name}/shape": list(param.shape),
                f"{model_name}/{name}/has_nan": torch.isnan(param).any().item(),
                f"{model_name}/{name}/nan_count": torch.isnan(param).sum().item(),
            }
            
            # 如果不是全是NaN才添加统计信息
            if not torch.isnan(param).all():
                param_stats.update({
                    f"{model_name}/{name}/min": param.min().item(),
                    f"{model_name}/{name}/max": param.max().item(),
                    f"{model_name}/{name}/mean": param.mean().item(),
                    f"{model_name}/{name}/std": param.std().item()
                })
            
            log_dict.update(param_stats)
        
        # 使用 Lightning 的 log_dict 方法记录
        self.log_dict(
            log_dict,
            prog_bar=False,  # 不在进度条显示
            logger=True,     # 记录到 logger
            # on_step=True,    # 每步记录
            on_epoch=True,   # 每轮记录
            sync_dist=self.sync_dist  # 分布式训练同步
        )
        return has_nan
    
    def test_step(self, batch, batch_idx):
        inputs = batch["image"]
        filename = batch["filename"]
        filename = filename[0]


        cond_cat = self.get_cond(batch, self.cond_type)

        reconstructions, _, z= self(inputs)
        
        cond_latent, cond_rec = self.conditional_encode(cond_cat)
        # self.check_model_params_detailed(self.cond_stage_model, "conditional_encode")
        # self.check_model_params_detailed(self.proj_head, "proj_head")
        # self.check_tensor(cond_latent,"cond_latent")
        # print(cond_latent)

        condloss= self.contrastive_loss(z, cond_latent, self.proj_head)
        rec_loss = F.mse_loss(cond_rec, inputs)
        cliploss = condloss + self.loss_ratio*rec_loss

        print("!!!!!CLIPLoss:", cliploss)


        cond_proj = self.proj_head(cond_latent)

        cond_base_rec = self.ae_model.decode(cond_proj) 
        # self.check_tensor(cond_base_rec, "cond_base_rec")
        cond_base_rec_loss = F.mse_loss(cond_base_rec, inputs)
        print("!!!!!COND_BASE_REC_Loss:", cond_base_rec_loss)

        self.img_saver(inputs, post_fix="origin_x", filename=str(filename)+"_origin_x")
        self.img_saver(reconstructions, post_fix="ae_rec", filename=str(filename)+"_ae_rec")

        # cond_z = self.condition_vit_encode(cond_cat)

        # cond_base_rec = self.ae_model.decode(cond_z)
        self.img_saver(cond_base_rec, post_fix="cond_base_rec", filename=str(filename)+"_cond_base_rec")
        self.img_saver(cond_rec, post_fix="cond_rec", filename=str(filename)+"_cond_rec")
        

    def configure_optimizers(self):
        lr = self.learning_rate
        # opt_ae = torch.optim.Adam(
        #     list(self.encoder.parameters())
        #     + list(self.decoder.parameters())
        #     + list(self.quant_conv.parameters())
        #     + list(self.post_quant_conv.parameters()),
        #     lr=lr,
        #     betas=(0.5, 0.9),
        # )
        # opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(), lr=lr, betas=(0.5, 0.9))
        opt_cond = torch.optim.Adam(self.cond_stage_model.parameters(), lr=lr, betas=(0.5, 0.9))
        return opt_cond

    def get_last_layer(self):
        return self.ae_model.decoder.conv_out.weight

    def to_image(x):
        x = torch.clamp(x, min=-1, max=1)
        x = (x + 1) * 127.5
        # x = x.squeeze(0).permute(1, 0, 2, 3)
        x = x.type(torch.uint8)
        x = x.cpu().numpy()
        return x


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(config):
    config = config["config"]
    import os
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))
    from dataset.monai_nii_dataset1 import AlignDataSet
    from torch.utils.data import DataLoader
    train_ds = AlignDataSet(config,split = "train")
    train_dl = DataLoader(
        dataset=train_ds,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=config.num_workers,
        batch_size=config.batch_size,
    )
    model = CLIPAE(save_path=config.hydra_path,config=config)
    trainer = pl.Trainer(max_epochs=10,devices=[0],fast_dev_run=False)
    trainer.fit(model, train_dl)
    # print(type(model.encoder))


if __name__ == "__main__":
    main()
    # cos = nn.CosineSimilarity(dim=1)
    # input1 = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float)
    # input2 = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float)
    # output = cos(input1, input2)
    # print(output)
