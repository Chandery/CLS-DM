import torch
import torch.nn as nn
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

# import pytorch_lightning as pl
import lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from torchvision.models import resnet50
from einops import rearrange, repeat
from contextlib import contextmanager
from functools import partial
from tqdm import tqdm
from torchvision.utils import make_grid, save_image
from torch import clamp
import SimpleITK as sitk
import torch.fft as fft
import torch.functional as F
from PIL import Image
from monai.transforms import SaveImage

from .lr_scheduler import LambdaLinearScheduler
from .unet import UNetModel


from .util import log_txt_as_img, exists, default, ismap, isimage, mean_flat, count_params, instantiate_from_config
from .ema import LitEma
from autoencoderkl.distributions import normal_kl, DiagonalGaussianDistribution
from autoencoderkl.autoencoder import AutoencoderKL, VQModelInterface, IdentityFirstStage
from .ddpm_utils import make_beta_schedule, extract_into_tensor, noise_like
from .ddim import DDIMSampler
from .Medicalnet.Vit import load_weight_for_vit_encoder, vit_encoder_b
from .condition_extractor import UnetEncoder
from .CLIPAutoEncoder.autoencoder import CLIPAE
from .dpm_solver.sampler import DPMSolverSampler


__conditioning_keys__ = {"concat": "c_concat", "crossattn": "c_crossattn", "adm": "y"}
class resnet_duplicate(nn.Module):
    def __init__(self,cond_type,ckpt_path):
        super().__init__()
        self.resnet = resnet50()
        # for name in self.resnet.named_modules():
        #     print(name)
        # self.resnet.layer4 = nn.Sequential()
        # self.resnet.avgpool = nn.Sequential()
        # self.resnet.fc = nn.Sequential()
        if cond_type=="pretrained_resnet":
            ckpt = torch.load(ckpt_path)
            self.resnet.load_state_dict(ckpt,strict=False)
            self.resnet.eval()
            for param in self.resnet.parameters():
                param.requires_grad = False
        
        # test_tensor = torch.randn((1,3,256,256))
        # print(test_tensor.shape)
        # for name,module in self.resnet._modules.items():
        #     print(name)
        #     print(module)
        #     test_tensor=module(test_tensor)
        #     print(test_tensor.shape)
        # for name in self.resnet.named_modules():
        #     print(name)
        self.fc = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 8, kernel_size=1, stride=1, padding=0),
        )
    def forward(self,x):
        for name,module in self.resnet._modules.items():
            if name=="layer4":
                break
            x=module(x)
        x = self.fc(x)
        _,_,h,w =  x.shape
        x = x.unsqueeze(-1)
        # x = torch.cat([x]*h,dim=-1)
        x = x.repeat(1,1,1,1,h)
        return x


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def uniform_on_device(r1, r2, shape, device):
    return (r1 - r2) * torch.rand(*shape, device=device) + r2


class DDPM(pl.LightningModule):
    # classic DDPM with Gaussian diffusion, in image space
    def __init__(
        self,
        unet_config,
        timesteps=500, # ! 1000
        beta_schedule="linear",
        loss_type="l2",
        ckpt_path=None,
        ignore_keys=[],
        load_only_unet=False,
        monitor="val/loss",
        use_ema=True,
        first_stage_key="image",
        image_size=256,
        channels=3,
        log_every_t=100,
        clip_denoised=True,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
        given_betas=None,
        original_elbo_weight=0.0,
        base_learning_rate=1e-5,
        v_posterior=0.0,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
        l_simple_weight=1.0,
        conditioning_key=None,
        parameterization="eps",  # all assuming fixed variance schedules
        scheduler_config=None,
        use_positional_encodings=False,
        learn_logvar=False,
        logvar_init=0.0,
        batch_size=1,
        pad_channel=8,
        root_path=None,
    ):
        super().__init__()
        assert parameterization in ["eps", "x0"], 'currently only supporting "eps" and "x0"'
        self.parameterization = parameterization
        print(f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode")
        # self.cond_stage_model = None
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.first_stage_key = first_stage_key
        self.image_size = image_size  # try conv?
        self.channels = channels
        self.use_positional_encodings = use_positional_encodings
        self.model = DiffusionWrapper(unet_config, conditioning_key)

        self.learning_rate = base_learning_rate
        self.batch_size = batch_size
        self.pad_channel = pad_channel
        self.root_path = root_path
        count_params(self.model, verbose=True)
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config

        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet)

        self.register_schedule(
            given_betas=given_betas,
            beta_schedule=beta_schedule,
            timesteps=timesteps,
            linear_start=linear_start,
            linear_end=linear_end,
            cosine_s=cosine_s,
        )

        self.loss_type = loss_type

        self.learn_logvar = learn_logvar
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,)).to(self.device)
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)

    def register_schedule(
        self, given_betas=None, beta_schedule="linear", timesteps=1000, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3
    ):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(
                beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s
            )
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, "alphas have to be defined for each timestep"

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod)))
        self.register_buffer("log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod)))
        self.register_buffer("sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod)))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1.0 - alphas_cumprod_prev) / (
            1.0 - alphas_cumprod
        ) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer("posterior_variance", to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer("posterior_log_variance_clipped", to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer("posterior_mean_coef1", to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)))
        self.register_buffer(
            "posterior_mean_coef2", to_torch((1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod))
        )

        if self.parameterization == "eps":
            lvlb_weights = self.betas**2 / (2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2.0 * 1 - torch.Tensor(alphas_cumprod))
        else:
            raise NotImplementedError("mu not supported")
        # TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer("lvlb_weights", lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = (
            self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(sd, strict=False)
        )
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool):
        model_out = self.model(x, t)
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        if clip_denoised:
            x_recon.clamp_(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, return_intermediates=False):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        intermediates = [img]
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc="Sampling t", total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), clip_denoised=self.clip_denoised)
            if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates.append(img)
        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(self, batch_size=16, return_intermediates=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size), return_intermediates=return_intermediates)

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == "l1":
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == "l2":
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction="none")
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def p_losses(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.model(x_noisy, t)

        loss_dict = {}
        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x_start
        else:
            raise NotImplementedError(f"Paramterization {self.parameterization} not yet supported")

        loss = self.get_loss(model_out, target, mean=False).mean(dim=[1, 2, 3])

        log_prefix = "train" if self.training else "val"

        loss_dict.update({f"{log_prefix}/loss_simple": loss.mean()})
        loss_simple = loss.mean() * self.l_simple_weight

        loss_vlb = (self.lvlb_weights[t] * loss).mean()
        loss_dict.update({f"{log_prefix}/loss_vlb": loss_vlb})

        loss = loss_simple + self.original_elbo_weight * loss_vlb

        loss_dict.update({f"{log_prefix}/loss": loss})

        return loss, loss_dict

    def forward(self, x, *args, **kwargs):
        # b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        # assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        return self.p_losses(x, t, *args, **kwargs)

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = rearrange(x, "b h w c -> b c h w")
        x = x.to(memory_format=torch.contiguous_format).float()
        return x

    def shared_step(self, batch):
        x = self.get_input(batch, self.first_stage_key)
        loss, loss_dict = self(x)
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        losses, _ = self.shared_step(batch)
        loss, loss_dict = losses

        loss_dict.update({"current_lr": self.optimizers().param_groups[0]["lr"]})
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        

        self.log("global_step", self.global_step, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]["lr"]
            self.log("lr_abs", lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _, loss_dict_no_ema = self.shared_step(batch)
        with self.ema_scope():
            _, loss_dict_ema = self.shared_step(batch)
            loss_dict_ema = {key + "_ema": loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    def _get_rows_from_list(self, samples):
        n_imgs_per_row = len(samples)
        denoise_grid = rearrange(samples, "n b c h w -> b n c h w")
        denoise_grid = rearrange(denoise_grid, "b n c h w -> (b n) c h w")
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=2, sample=True, return_keys=None, **kwargs):
        log = dict()
        x = self.get_input(batch, self.first_stage_key)
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        x = x.to(self.device)[:N]
        log["inputs"] = x

        # get diffusion row
        diffusion_row = list()
        x_start = x[:n_row]

        for t in range(self.num_timesteps):
            if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                t = repeat(torch.tensor([t]), "1 -> b", b=n_row)
                t = t.to(self.device).long()
                noise = torch.randn_like(x_start)
                x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
                diffusion_row.append(x_noisy)

        log["diffusion_row"] = self._get_rows_from_list(diffusion_row)

        if sample:
            # get denoise row
            with self.ema_scope("Plotting"):
                samples, denoise_row = self.sample(batch_size=N, return_intermediates=True)

            log["samples"] = samples
            log["denoise_row"] = self._get_rows_from_list(denoise_row)

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.learn_logvar:
            params = params + [self.logvar]
        opt = torch.optim.AdamW(params, lr=lr)
        return opt


class LatentDiffusion(DDPM):
    """main class"""

    def __init__(
        self,
        config,
        first_stage_config,
        cond_stage_config=None,
        num_timesteps_cond=None,
        cond_stage_key="image",
        cond_stage_trainable=False,
        concat_mode=True,
        cond_stage_forward=None,
        conditioning_key=None,
        scale_factor=1.0,
        scale_by_std=False,
        high_low_mode=False,
        cond_nums=[1,2],
        dpm_type="ddim",
        *args,
        **kwargs,
    ):
        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std
        self.config = config
        assert self.num_timesteps_cond <= kwargs["timesteps"]
        # for backwards compatibility after implementation of DiffusionWrapper
        if conditioning_key is None:
            conditioning_key = "concat" if concat_mode else "crossattn"
        if cond_stage_config == "__is_unconditional__":
            conditioning_key = None
        # * condition_key should be concat here
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        super().__init__(conditioning_key=conditioning_key, *args, **kwargs)
        self.concat_mode = concat_mode
        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key
        try:
            self.num_downs = len(first_stage_config.params.ddconfig.ch_mult) - 1
        except:
            self.num_downs = 0
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer("scale_factor", torch.tensor(scale_factor))
        self.instantiate_first_stage_and_cond_stage(first_stage_config, self.config)  # * load clipae
        # ! cond_stage_model has initialized in first_stage_model
        # self.instantiate_cond_stage(cond_stage_config)  # * load cond stage model 
        self.cond_stage_forward = cond_stage_forward  # * false
        self.clip_denoised = False  # * false?
        self.bbox_tokenizer = None
        self.high_low_mode = high_low_mode
        self.cond_nums = cond_nums
        self.dpm_type = dpm_type

        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)
            self.restarted_from_ckpt = True

    def make_cond_schedule(
        self,
    ):
        self.cond_ids = torch.full(size=(self.num_timesteps,), fill_value=self.num_timesteps - 1, dtype=torch.long)
        ids = torch.round(torch.linspace(0, self.num_timesteps - 1, self.num_timesteps_cond)).long()
        self.cond_ids[: self.num_timesteps_cond] = ids

    def lowpass_torch(self, input, limit):
        pass1 = torch.abs(fft.rfftfreq(input.shape[-1])) < limit
        pass2 = torch.abs(fft.fftfreq(input.shape[-2])) < limit
        kernel = torch.outer(pass2, pass1).to(input)
        fft_input = fft.rfftn(input)
        return fft.irfftn(fft_input * kernel, s=input.shape[-3:])

    def highpass_torch(self, input, limit):
        pass1 = torch.abs(fft.rfftfreq(input.shape[-1])) > limit
        pass2 = torch.abs(fft.fftfreq(input.shape[-2])) > limit
        kernel = torch.outer(pass2, pass1).to(input)
        fft_input = fft.rfftn(input)
        return fft.irfftn(fft_input * kernel, s=input.shape[-3:])

    def _high_low_loss(self, rec, target, low_limit, high_limit):
        rec_low = self.lowpass_torch(rec, low_limit)
        target_low = self.lowpass_torch(target, low_limit)

        rec_high = self.highpass_torch(rec, high_limit)
        target_high = self.highpass_torch(target, high_limit)

        return F.mse_loss(rec_low, target_low) + F.mse_loss(rec_high, target_high)

    # @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx):
        # only for very first batch
        if (
            self.scale_by_std
            and self.current_epoch == 0
            and self.global_step == 0
            and batch_idx == 0
            and not self.restarted_from_ckpt
            and self.global_rank == 0
        ):
            assert self.scale_factor == 1.0, "rather not use custom rescaling and std-rescaling simultaneously"
            # set rescale weight to 1./std of encodings
            print("### USING STD-RESCALING ###")
            # x = super().get_input(batch, self.first_stage_key)
            # x, _ = batch
            x = batch["image"]
            x = x.to(self.device)
            encoder_posterior = self.encode_first_stage(x)
            # print(type(encoder_posterior))
            z = self.get_first_stage_encoding(encoder_posterior).detach()
            del self.scale_factor
            self.register_buffer("scale_factor", 1.0 / z.flatten().std())
            print(f"setting self.scale_factor to {self.scale_factor}")
            print("### USING STD-RESCALING ###")

    def register_schedule(
        self, given_betas=None, beta_schedule="linear", timesteps=1000, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3
    ):
        super().register_schedule(given_betas, beta_schedule, timesteps, linear_start, linear_end, cosine_s)

        self.shorten_cond_schedule = self.num_timesteps_cond > 1
        if self.shorten_cond_schedule:
            self.make_cond_schedule()

    def instantiate_first_stage_and_cond_stage(self, config, global_config):
        model = CLIPAE(save_path=self.config.hydra_path,config=config)
        
        if config.cond_flag == "clip":
            model.init_from_ckpt(config.clipae_ckpt)
            self.first_stage_model = model.eval()
            self.first_stage_model.train = disabled_train
            for param in self.first_stage_model.parameters():
                param.requires_grad = False
        if config.cond_flag == "unclip":
            self.first_stage_model = model.train()

        # ? init cond_stage_model
        # self.cond_stage_model = self.first_stage_model.cond_stage_model # ? clip cond_stage_model
        # self.clipae_model = self.first_stage_model# ? clip 
        # self.first_stage_model = self.first_stage_model.ae_model # ? clip ae_model
    @property
    def cond_stage_model(self):
        return self.first_stage_model.cond_stage_model
    
    def instantiate_cond_stage(self, config):
        # ! useless

        if config.cond_type=="pretrained_vit":
            model = vit_encoder_b()
            model.fc = nn.Sequential(
                nn.Conv2d(768, 512, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(128, 8, kernel_size=1, stride=1, padding=0),
            )
            pretrained_weights = load_weight_for_vit_encoder("lvm-med-vit", config.cond_ckpt_path)
            model.load_state_dict(pretrained_weights, strict=False)
            model.eval()
        if config.cond_type=="vit":
            model = vit_encoder_b()
            model.fc = nn.Sequential(
                nn.Conv2d(768, 512, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(128, 8, kernel_size=1, stride=1, padding=0),
            )
        elif config.cond_type=="pretrained_resnet":
            model = resnet_duplicate(config.cond_type,config.cond_ckpt_path)
            print(f"load weights from {config.cond_ckpt_path}")
            print("Using Pretrained Resnet(Only train fc layers)")
        elif config.cond_type=="resnet":
            model = resnet_duplicate(config.cond_type,config.cond_ckpt_path)
            print(f"load weights from {config.cond_ckpt_path}")
            print("Training Resnet from scratch")
        elif config.cond_type=="unet_encoder":
            model = UnetEncoder(in_channels=3)

        # self.cond_stage_model = model  # ! useless

    def _get_denoise_row_from_list(self, samples, desc="", force_no_decoder_quantization=False):
        denoise_row = []
        for zd in tqdm(samples, desc=desc):
            denoise_row.append(self.decode_first_stage(zd.to(self.device), force_not_quantize=force_no_decoder_quantization))
        n_imgs_per_row = len(denoise_row)
        denoise_row = torch.stack(denoise_row)  # n_log_step, n_row, C, H, W
        denoise_grid = rearrange(denoise_row, "n b c h w -> b n c h w")
        denoise_grid = rearrange(denoise_grid, "b n c h w -> (b n) c h w")
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid

    def get_first_stage_encoding(self, encoder_posterior):
        # print(type(encoder_posterior))
        # print(DiagonalGaussianDistribution)
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            # print("!!!!!fuck")
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z

    def get_learned_conditioning(self, c):
        if self.cond_stage_forward is None:
            if hasattr(self.first_stage_model, "encode") and callable(self.first_stage_model.encode):
                c = self.first_stage_model.encode(c)
                if isinstance(c, DiagonalGaussianDistribution):
                    c = c.mode()
            else:
                c = self.first_stage_model(c)
        else:
            assert hasattr(self.first_stage_model, self.cond_stage_forward)
            c = getattr(self.first_stage_model, self.cond_stage_forward)(c)
        return c

    def condition_vit_encode(self, cond):
        """
        using vit backbone to encode conditioning x-ray imgs.
        backbone checkpoint from https://github.com/duyhominhnguyen/LVM-Med
        input: (1,2,256,256,256)
        output: (1,4,16,16,16) match the latent code z.
        """
        # * repeat second channel
        cond, cond_rec = self.cond_stage_model(cond)
        return cond

    def meshgrid(self, h, w):
        y = torch.arange(0, h).view(h, 1, 1).repeat(1, w, 1)
        x = torch.arange(0, w).view(1, w, 1).repeat(h, 1, 1)

        arr = torch.cat([y, x], dim=-1)
        return arr

    def delta_border(self, h, w):
        """
        :param h: height
        :param w: width
        :return: normalized distance to image border,
         wtith min distance = 0 at border and max dist = 0.5 at image center
        """
        lower_right_corner = torch.tensor([h - 1, w - 1]).view(1, 1, 2)
        arr = self.meshgrid(h, w) / lower_right_corner
        dist_left_up = torch.min(arr, dim=-1, keepdims=True)[0]
        dist_right_down = torch.min(1 - arr, dim=-1, keepdims=True)[0]
        edge_dist = torch.min(torch.cat([dist_left_up, dist_right_down], dim=-1), dim=-1)[0]
        return edge_dist

    def get_weighting(self, h, w, Ly, Lx, device):
        weighting = self.delta_border(h, w)
        weighting = torch.clip(
            weighting,
            self.split_input_params["clip_min_weight"],
            self.split_input_params["clip_max_weight"],
        )
        weighting = weighting.view(1, h * w, 1).repeat(1, 1, Ly * Lx).to(device)

        if self.split_input_params["tie_braker"]:
            L_weighting = self.delta_border(Ly, Lx)
            L_weighting = torch.clip(
                L_weighting, self.split_input_params["clip_min_tie_weight"], self.split_input_params["clip_max_tie_weight"]
            )

            L_weighting = L_weighting.view(1, 1, Ly * Lx).to(device)
            weighting = weighting * L_weighting
        return weighting

    def get_fold_unfold(self, x, kernel_size, stride, uf=1, df=1):  # todo load once not every time, shorten code
        """
        :param x: img of size (bs, c, h, w)
        :return: n img crops of size (n, bs, c, kernel_size[0], kernel_size[1])
        """
        bs, nc, h, w = x.shape

        # number of crops in image
        Ly = (h - kernel_size[0]) // stride[0] + 1
        Lx = (w - kernel_size[1]) // stride[1] + 1

        if uf == 1 and df == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)

            fold = torch.nn.Fold(output_size=x.shape[2:], **fold_params)

            weighting = self.get_weighting(kernel_size[0], kernel_size[1], Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h, w)  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0], kernel_size[1], Ly * Lx))

        elif uf > 1 and df == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)

            fold_params2 = dict(
                kernel_size=(kernel_size[0] * uf, kernel_size[0] * uf),
                dilation=1,
                padding=0,
                stride=(stride[0] * uf, stride[1] * uf),
            )
            fold = torch.nn.Fold(output_size=(x.shape[2] * uf, x.shape[3] * uf), **fold_params2)

            weighting = self.get_weighting(kernel_size[0] * uf, kernel_size[1] * uf, Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h * uf, w * uf)  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0] * uf, kernel_size[1] * uf, Ly * Lx))

        elif df > 1 and uf == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)

            fold_params2 = dict(
                kernel_size=(kernel_size[0] // df, kernel_size[0] // df),
                dilation=1,
                padding=0,
                stride=(stride[0] // df, stride[1] // df),
            )
            fold = torch.nn.Fold(output_size=(x.shape[2] // df, x.shape[3] // df), **fold_params2)

            weighting = self.get_weighting(kernel_size[0] // df, kernel_size[1] // df, Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h // df, w // df)  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0] // df, kernel_size[1] // df, Ly * Lx))

        else:
            raise NotImplementedError

        return fold, unfold, normalization, weighting

    @torch.no_grad()
    def get_input(
        self,
        batch,
        k,
        return_first_stage_outputs=False,
        force_c_encode=False,
        cond_key=None,
        return_original_cond=False,
        bs=None,
    ):
        # x = super().get_input(batch, k)
        # x, cond1, cond2 = batch["image"].as_tensor(), batch["cond1"].as_tensor(), batch["cond2"].as_tensor()
        # x = batch["image"].as_tensor() # ? monai
        x = torch.as_tensor(batch["image"])

        if return_original_cond:
            cond = []
            if 1 in self.cond_nums:
                cond1 = torch.as_tensor(batch["cond1"])
            else:
                cond1 = None
            if 2 in self.cond_nums:
                cond2 = torch.as_tensor(batch["cond2"])
            else:
                cond2 = None
            if 3 in self.cond_nums:
                cond3 = torch.as_tensor(batch["cond3"])
            else:
                cond3 = None
            cond = [cond1, cond2, cond3]
        
        cond_cat = self.first_stage_model.get_cond(batch, self.first_stage_model.cond_type)

        if bs is not None:
            x = x[:bs]
        x = x.to(self.device)
        encoder_posterior = self.encode_first_stage(x)
        z = self.get_first_stage_encoding(encoder_posterior).detach()  # *  input image -> first stage -> sample -> z
        

        # if self.high_low_mode:
        #     high_c = self.highpass_torch(cond, limit=0.04)
        #     high_c = self.get_learned_conditioning(high_c)
        #     low_c = self.lowpass_torch(cond, limit=0.04)
        #     low_c = self.get_learned_conditioning(low_c)

        #     c = self.get_learned_conditioning(cond)
        #     c = torch.concat([c, high_c, low_c], dim=1)
        # else:
        #     # c = self.get_learned_conditioning(cond)
        #     cond_cat = []
        #     if 1 in self.cond_nums:
        #         cond1 = self.condition_vit_encode(cond1)
        #         # print(cond1.shape) # 2,8,8,8,8
        #         cond_cat.append(cond1)
        #     if 2 in self.cond_nums:
        #         cond2 = self.condition_vit_encode(cond2)
        #         cond_cat.append(cond2)
        #     if 3 in self.cond_nums:
        #         cond3 = self.condition_vit_encode(cond3)
        #         cond_cat.append(cond3)

        #     c = torch.cat(cond_cat, dim=1)
        #     # print(c.shape) # 2,16,8,8,8

        c = self.condition_vit_encode(cond_cat)

        out = [z, c]
        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.extend([x, xrec])
        if return_original_cond:
            # out.append(cond)
            out.extend(cond)
        return out

    @torch.no_grad()
    def decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, "b h w c -> b c h w").contiguous()

        z = 1.0 / self.scale_factor * z

        if hasattr(self, "split_input_params"):
            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks"]  # eg. (128, 128)
                stride = self.split_input_params["stride"]  # eg. (64, 64)
                uf = self.split_input_params["vqf"]
                bs, nc, h, w = z.shape
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("reducing Kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing stride")

                fold, unfold, normalization, weighting = self.get_fold_unfold(z, ks, stride, uf=uf)

                z = unfold(z)  # (bn, nc * prod(**ks), L)
                # 1. Reshape to img shape
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                # 2. apply model loop over last dim
                if isinstance(self.first_stage_model, VQModelInterface):
                    output_list = [
                        self.first_stage_model.decode(
                            z[:, :, :, :, i], force_not_quantize=predict_cids or force_not_quantize
                        )
                        for i in range(z.shape[-1])
                    ]
                else:
                    output_list = [self.first_stage_model.decode(z[:, :, :, :, i]) for i in range(z.shape[-1])]

                o = torch.stack(output_list, axis=-1)  # # (bn, nc, ks[0], ks[1], L)
                o = o * weighting
                # Reverse 1. reshape to img shape
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # stitch crops together
                decoded = fold(o)
                decoded = decoded / normalization  # norm is shape (1, 1, h, w)
                return decoded
            else:
                if isinstance(self.first_stage_model, VQModelInterface):
                    return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
                else:
                    return self.first_stage_model.decode(z)

        else:
            if isinstance(self.first_stage_model, VQModelInterface):
                return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
            else:
                return self.first_stage_model.decode(z)

    # same as above but without decorator
    def differentiable_decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, "b h w c -> b c h w").contiguous()

        z = 1.0 / self.scale_factor * z

        if hasattr(self, "split_input_params"):
            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks"]  # eg. (128, 128)
                stride = self.split_input_params["stride"]  # eg. (64, 64)
                uf = self.split_input_params["vqf"]
                bs, nc, h, w = z.shape
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("reducing Kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing stride")

                fold, unfold, normalization, weighting = self.get_fold_unfold(z, ks, stride, uf=uf)

                z = unfold(z)  # (bn, nc * prod(**ks), L)
                # 1. Reshape to img shape
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                # 2. apply model loop over last dim
                if isinstance(self.first_stage_model, VQModelInterface):
                    output_list = [
                        self.first_stage_model.decode(
                            z[:, :, :, :, i], force_not_quantize=predict_cids or force_not_quantize
                        )
                        for i in range(z.shape[-1])
                    ]
                else:
                    output_list = [self.first_stage_model.decode(z[:, :, :, :, i]) for i in range(z.shape[-1])]

                o = torch.stack(output_list, axis=-1)  # # (bn, nc, ks[0], ks[1], L)
                o = o * weighting
                # Reverse 1. reshape to img shape
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # stitch crops together
                decoded = fold(o)
                decoded = decoded / normalization  # norm is shape (1, 1, h, w)
                return decoded
            else:
                if isinstance(self.first_stage_model, VQModelInterface):
                    return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
                else:
                    return self.first_stage_model.decode(z)

        else:
            if isinstance(self.first_stage_model, VQModelInterface):
                return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
            else:
                return self.first_stage_model.decode(z)

    @torch.no_grad()
    def encode_first_stage(self, x):
        if hasattr(self, "split_input_params"):
            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks"]  # eg. (128, 128)
                stride = self.split_input_params["stride"]  # eg. (64, 64)
                df = self.split_input_params["vqf"]
                self.split_input_params["original_image_size"] = x.shape[-2:]
                bs, nc, h, w = x.shape
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("reducing Kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing stride")

                fold, unfold, normalization, weighting = self.get_fold_unfold(x, ks, stride, df=df)
                z = unfold(x)  # (bn, nc * prod(**ks), L)
                # Reshape to img shape
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                output_list = [self.first_stage_model.encode(z[:, :, :, :, i]) for i in range(z.shape[-1])]

                o = torch.stack(output_list, axis=-1)
                o = o * weighting

                # Reverse reshape to img shape
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # stitch crops together
                decoded = fold(o)
                decoded = decoded / normalization
                return decoded

            else:
                return self.first_stage_model.encode(x)
        else:
            # print("!!!! to encode !!!!")
            return self.first_stage_model.encode(x)

    def shared_step(self, batch, **kwargs):
        x, c = self.get_input(batch, self.first_stage_key)  # * x: fisrt_stage gauss sample  ,c origin image
        loss = self(x, c)
        return loss, c

    def forward(self, x, c, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long() # ? 一次有batch_size个t
        # if self.model.conditioning_key is not None:
        #     assert c is not None
        #     if self.cond_stage_trainable:
        #         c = self.get_learned_conditioning(c)
        #     if self.shorten_cond_schedule:  # TODO: drop this option
        #         tc = self.cond_ids[t].to(self.device)
        #         c = self.q_sample(x_start=c, t=tc, noise=torch.randn_like(c.float()))  # * c -> c_tc(noisy)
        return self.p_losses(x, c, t, *args, **kwargs)

    def _rescale_annotations(self, bboxes, crop_coordinates):  # TODO: move to dataset
        def rescale_bbox(bbox):
            x0 = clamp((bbox[0] - crop_coordinates[0]) / crop_coordinates[2])
            y0 = clamp((bbox[1] - crop_coordinates[1]) / crop_coordinates[3])
            w = min(bbox[2] / crop_coordinates[2], 1 - x0)
            h = min(bbox[3] / crop_coordinates[3], 1 - y0)
            return x0, y0, w, h

        return [rescale_bbox(b) for b in bboxes]

    def apply_model(self, x_noisy, t, cond, return_ids=False):
        if isinstance(cond, dict):
            # hybrid case, cond is exptected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = "c_concat" if self.model.conditioning_key == "concat" else "c_crossattn"
            cond = {key: cond}

        if hasattr(self, "split_input_params"):
            assert len(cond) == 1  # todo can only deal with one conditioning atm
            assert not return_ids
            ks = self.split_input_params["ks"]  # eg. (128, 128)
            stride = self.split_input_params["stride"]  # eg. (64, 64)

            h, w = x_noisy.shape[-2:]

            fold, unfold, normalization, weighting = self.get_fold_unfold(x_noisy, ks, stride)

            z = unfold(x_noisy)  # (bn, nc * prod(**ks), L)
            # Reshape to img shape
            z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )
            z_list = [z[:, :, :, :, i] for i in range(z.shape[-1])]

            if (
                self.cond_stage_key in ["image", "LR_image", "segmentation", "bbox_img"] and self.model.conditioning_key
            ):  # todo check for completeness
                c_key = next(iter(cond.keys()))  # get key
                c = next(iter(cond.values()))  # get value
                assert len(c) == 1  # todo extend to list with more than one elem
                c = c[0]  # get element

                c = unfold(c)
                c = c.view((c.shape[0], -1, ks[0], ks[1], c.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                cond_list = [{c_key: [c[:, :, :, :, i]]} for i in range(c.shape[-1])]

            elif self.cond_stage_key == "coordinates_bbox":
                assert "original_image_size" in self.split_input_params, "BoudingBoxRescaling is missing original_image_size"

                # assuming padding of unfold is always 0 and its dilation is always 1
                n_patches_per_row = int((w - ks[0]) / stride[0] + 1)
                full_img_h, full_img_w = self.split_input_params["original_image_size"]
                # as we are operating on latents, we need the factor from the original image size to the
                # spatial latent size to properly rescale the crops for regenerating the bbox annotations
                num_downs = self.first_stage_model.encoder.num_resolutions - 1
                rescale_latent = 2 ** (num_downs)

                # get top left postions of patches as conforming for the bbbox tokenizer, therefore we
                # need to rescale the tl patch coordinates to be in between (0,1)
                tl_patch_coordinates = [
                    (
                        rescale_latent * stride[0] * (patch_nr % n_patches_per_row) / full_img_w,
                        rescale_latent * stride[1] * (patch_nr // n_patches_per_row) / full_img_h,
                    )
                    for patch_nr in range(z.shape[-1])
                ]

                # patch_limits are tl_coord, width and height coordinates as (x_tl, y_tl, h, w)
                patch_limits = [
                    (x_tl, y_tl, rescale_latent * ks[0] / full_img_w, rescale_latent * ks[1] / full_img_h)
                    for x_tl, y_tl in tl_patch_coordinates
                ]
                # patch_values = [(np.arange(x_tl,min(x_tl+ks, 1.)),np.arange(y_tl,min(y_tl+ks, 1.))) for x_tl, y_tl in tl_patch_coordinates]

                # tokenize crop coordinates for the bounding boxes of the respective patches
                patch_limits_tknzd = [
                    torch.LongTensor(self.bbox_tokenizer._crop_encoder(bbox))[None].to(self.device) for bbox in patch_limits
                ]  # list of length l with tensors of shape (1, 2)
                print(patch_limits_tknzd[0].shape)
                # cut tknzd crop position from conditioning
                assert isinstance(cond, dict), "cond must be dict to be fed into model"
                cut_cond = cond["c_crossattn"][0][..., :-2].to(self.device)
                print(cut_cond.shape)

                adapted_cond = torch.stack([torch.cat([cut_cond, p], dim=1) for p in patch_limits_tknzd])
                adapted_cond = rearrange(adapted_cond, "l b n -> (l b) n")
                print(adapted_cond.shape)
                adapted_cond = self.get_learned_conditioning(adapted_cond)
                print(adapted_cond.shape)
                adapted_cond = rearrange(adapted_cond, "(l b) n d -> l b n d", l=z.shape[-1])
                print(adapted_cond.shape)

                cond_list = [{"c_crossattn": [e]} for e in adapted_cond]

            else:
                cond_list = [cond for i in range(z.shape[-1])]  # Todo make this more efficient

            # apply model by loop over crops
            output_list = [self.model(z_list[i], t, **cond_list[i]) for i in range(z.shape[-1])]
            assert not isinstance(
                output_list[0], tuple
            )  # todo cant deal with multiple model outputs check this never happens

            o = torch.stack(output_list, axis=-1)
            o = o * weighting
            # Reverse reshape to img shape
            o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
            # stitch crops together
            x_recon = fold(o) / normalization

        else:
            x_recon = self.model(x_noisy, t, **cond)

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart) / extract_into_tensor(
            self.sqrt_recipm1_alphas_cumprod, t, x_t.shape
        )

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.
        This term can't be optimized, as it only depends on the encoder.
        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = torch.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0)
        return mean_flat(kl_prior) / np.log(2.0)

    def p_losses(self, x_start, cond, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))  # * [64,64,4,4,4]
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)  # * x -> x_t (noisey)
        model_output = self.apply_model(x_noisy, t, cond)

        loss_dict = {}
        prefix = "train" if self.training else "val"

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        else:
            raise NotImplementedError()

        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3, 4])
        loss_dict.update({f"{prefix}/loss_simple": loss_simple.mean()})

        self.logvar = self.logvar.to(self.device)
        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.learn_logvar:
            loss_dict.update({f"{prefix}/loss_gamma": loss.mean()})
            loss_dict.update({"logvar": self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3, 4))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f"{prefix}/loss_vlb": loss_vlb})
        loss += self.original_elbo_weight * loss_vlb
        loss_dict.update({f"{prefix}/loss": loss})

        return loss, loss_dict

    def p_mean_variance(
        self,
        x,
        c,
        t,
        clip_denoised: bool,
        return_codebook_ids=False,
        quantize_denoised=False,
        return_x0=False,
        score_corrector=None,
        corrector_kwargs=None,
    ):
        t_in = t
        model_out = self.apply_model(x, t_in, c, return_ids=return_codebook_ids)

        if score_corrector is not None:
            assert self.parameterization == "eps"
            model_out = score_corrector.modify_score(self, model_out, x, t, c, **corrector_kwargs)

        if return_codebook_ids:
            model_out, logits = model_out

        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        else:
            raise NotImplementedError()

        if clip_denoised:
            x_recon.clamp_(-1.0, 1.0)
        if quantize_denoised:
            x_recon, _, [_, _, indices] = self.first_stage_model.quantize(x_recon)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        if return_codebook_ids:
            return model_mean, posterior_variance, posterior_log_variance, logits
        elif return_x0:
            return model_mean, posterior_variance, posterior_log_variance, x_recon
        else:
            return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(
        self,
        x,
        c,
        t,
        clip_denoised=False,
        repeat_noise=False,
        return_codebook_ids=False,
        quantize_denoised=False,
        return_x0=False,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
    ):
        b, *_, device = *x.shape, x.device
        outputs = self.p_mean_variance(
            x=x,
            c=c,
            t=t,
            clip_denoised=clip_denoised,
            return_codebook_ids=return_codebook_ids,
            quantize_denoised=quantize_denoised,
            return_x0=return_x0,
            score_corrector=score_corrector,
            corrector_kwargs=corrector_kwargs,
        )
        if return_codebook_ids:
            raise DeprecationWarning("Support dropped.")
            model_mean, _, model_log_variance, logits = outputs
        elif return_x0:
            model_mean, _, model_log_variance, x0 = outputs
        else:
            model_mean, _, model_log_variance = outputs

        noise = noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.0:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        if return_codebook_ids:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, logits.argmax(dim=1)
        if return_x0:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x0
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def progressive_denoising(
        self,
        cond,
        shape,
        verbose=True,
        callback=None,
        quantize_denoised=False,
        img_callback=None,
        mask=None,
        x0=None,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        batch_size=None,
        x_T=None,
        start_T=None,
        log_every_t=None,
    ):
        if not log_every_t:
            log_every_t = self.log_every_t
        timesteps = self.num_timesteps
        if batch_size is not None:
            b = batch_size if batch_size is not None else shape[0]
            shape = [batch_size] + list(shape)
        else:
            b = batch_size = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=self.device)
        else:
            img = x_T
        intermediates = []
        if cond is not None:
            if isinstance(cond, dict):
                cond = {
                    key: cond[key][:batch_size]
                    if not isinstance(cond[key], list)
                    else list(map(lambda x: x[:batch_size], cond[key]))
                    for key in cond
                }
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = (
            tqdm(reversed(range(0, timesteps)), desc="Progressive Generation", total=timesteps)
            if verbose
            else reversed(range(0, timesteps))
        )
        if type(temperature) == float:
            temperature = [temperature] * timesteps

        for i in iterator:
            ts = torch.full((b,), i, device=self.device, dtype=torch.long)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != "hybrid"
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            img, x0_partial = self.p_sample(
                img,
                cond,
                ts,
                clip_denoised=self.clip_denoised,
                quantize_denoised=quantize_denoised,
                return_x0=True,
                temperature=temperature[i],
                noise_dropout=noise_dropout,
                score_corrector=score_corrector,
                corrector_kwargs=corrector_kwargs,
            )
            if mask is not None:
                assert x0 is not None
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1.0 - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(x0_partial)
            if callback:
                callback(i)
            if img_callback:
                img_callback(img, i)
        return img, intermediates

    @torch.no_grad()
    def p_sample_loop(
        self,
        cond,
        shape,
        return_intermediates=False,
        x_T=None,
        verbose=True,
        callback=None,
        timesteps=None,
        quantize_denoised=False,
        mask=None,
        x0=None,
        img_callback=None,
        start_T=None,
        log_every_t=None,
    ):
        if not log_every_t:
            log_every_t = self.log_every_t
        device = self.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        intermediates = [img]
        if timesteps is None:
            timesteps = self.num_timesteps

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = (
            tqdm(reversed(range(0, timesteps)), desc="Sampling t", total=timesteps)
            if verbose
            else reversed(range(0, timesteps))
        )

        if mask is not None:
            assert x0 is not None
            assert x0.shape[2:3] == mask.shape[2:3]  # spatial size has to match

        for i in iterator:
            ts = torch.full((b,), i, device=device, dtype=torch.long)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != "hybrid"
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            img = self.p_sample(img, cond, ts, clip_denoised=self.clip_denoised, quantize_denoised=quantize_denoised)
            if mask is not None:
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1.0 - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(img)
            if callback:
                callback(i)
            if img_callback:
                img_callback(img, i)

        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(
        self,
        cond,
        batch_size=16,
        return_intermediates=False,
        x_T=None,
        verbose=True,
        timesteps=None,
        quantize_denoised=False,
        mask=None,
        x0=None,
        shape=None,
        **kwargs,
    ):
        if shape is None:
            shape = (batch_size, self.channels, self.image_size, self.image_size)
        if cond is not None:
            if isinstance(cond, dict):
                cond = {
                    key: cond[key][:batch_size]
                    if not isinstance(cond[key], list)
                    else list(map(lambda x: x[:batch_size], cond[key]))
                    for key in cond
                }
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]
        return self.p_sample_loop(
            cond,
            shape,
            return_intermediates=return_intermediates,
            x_T=x_T,
            verbose=verbose,
            timesteps=timesteps,
            quantize_denoised=quantize_denoised,
            mask=mask,
            x0=x0,
        )

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        if ddim:
            ddim_sampler = DDIMSampler(self)
            shape = (self.channels, self.image_size, self.image_size)
            samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)

        else:
            samples, intermediates = self.sample(cond=cond, batch_size=batch_size, return_intermediates=True, **kwargs)

        return samples, intermediates

    def validation_step(self, batch, batch_idx):
        # _, loss_dict_no_ema, cond = self.shared_step(batch)
        # if self.current_epoch % 10 == 0 and batch_idx == 1:
        if batch_idx == 1:
            val_losses, cond = self.shared_step(batch)
            _, loss_dict_no_ema = val_losses
            # with self.ema_scope():
            #     # _, loss_dict_ema = self.shared_step(batch)
            #     val_losses, _ = self.shared_step(batch)
            #     _, loss_dict_ema = val_losses
            #     loss_dict_ema = {key + "_ema": loss_dict_ema[key] for key in loss_dict_ema}
            self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
            # self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
            # if self.current_epoch % 100 == 0 and batch_idx == 1:
            val_losses, cond = self.shared_step(batch)
            shape = (self.batch_size, self.channels, self.image_size, self.image_size, self.image_size)
            assert self.dpm_type in ["ddim", "dpm-solver"], "dpm_type must be either 'ddim' or 'dpm-solver'"
            if self.dpm_type == "ddim":
                ddim_sampler = DDIMSampler(self)
                cond_z, _ = ddim_sampler.sample(50, batch_size=self.batch_size, shape=shape, conditioning=cond, verbose=False)
                print(f"ddim shape :{shape}")
            elif self.dpm_type == "dpm-solver":
                dpm_sampler = DPMSolverSampler(self)
                print(f"dpm shape :{shape}") 
                cond_z, _ = dpm_sampler.sample(S=10, batch_size=self.batch_size, shape=shape, conditioning=cond, verbose=False)

            reconstructions = self.decode_first_stage(cond_z)
            reconstructions = torch.clamp(reconstructions, min=-1, max=1)
            reconstructions = (reconstructions + 1) * 127.5
            rec = reconstructions[0]

            reconstructions = reconstructions[0].permute(1, 0, 2, 3)
            reconstructions = reconstructions.type(torch.uint8)
            grid = make_grid(reconstructions)
            self.logger.experiment.add_image("val_rec", grid, self.global_step)

            x = batch["image"]

            x = torch.clamp(x, min=-1, max=1)
            x = (x + 1) * 127.5
            gt = x[0]
            x = x[0].permute(1, 0, 2, 3)
            x = x.type(torch.uint8)
            grid = make_grid(x)
            self.logger.experiment.add_image("val_gt", grid, self.global_step)

            from scripts.metrics import Structural_Similarity
            gt = gt / 255 * self.config["CT_MIN_MAX"][1]
            rec = rec / 255 * self.config["CT_MIN_MAX"][1]
            _, _, _, ssim_avg = Structural_Similarity(gt, rec, PIXEL_MAX=self.config["DATA_MIN_MAX"][1])
            self.log("val/ssim", ssim_avg)
            print(f"current epoch {self.current_epoch} aims a validation step,\n ssim: {ssim_avg}\n loss_val: {loss_dict_no_ema['val/loss_simple']}")

            # from skimage.metrics import structural_similarity as ssim
            # self.log("val_ssim", ssim(rec, gt, data_range=gt.max()-gt.min()))

    def img_saver(self, img, post_fix, i_type=".nii", meta_data=None,filename=None, **kwargs):
        """
        save img to self.root_path with post_fix

        Args:
            img (torch.Tensor): [description]
            post_fix (str): [description]
            type (str, optional): [description]. Defaults to "nii".
            meta_data ([type], optional): [description]. Defaults to None.
        """
        # if hasattr(img, "meta") and meta_data is None:
        #     meta_data = img.meta
        # elif meta_data is None:
        #     print("img dosen't has meta attribution use `None` as meta_data")
        # else:
        #     print("use the input meta data")

        assert i_type in [".nii", ".nii.gz", ".jpg"], "Only .nii or .jpg suffix file supported now"
        # assert post_fix in ["origin_x", "ae_rec", "xray_left", "xray_right", "rec"], "unsupported post_fix"

        img = img.squeeze(0)
        # if post_fix in ["ae_rec","rec"]:
        #     MAX = torch.max(img)
        #     MIN = torch.min(img)
        #     img = 2*(img-MAX)/(MAX-MIN)-1
        # else:
        img = torch.clamp(img, min=-1, max=1)
        img = (img + 1) * 127.5
        writer = "NibabelWriter" if "nii" in i_type else "PILWriter"
        out_ext = ".nii.gz" if "nii" in i_type else ".jpg"

        # print("ROOTPATH=",self.root_path)
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
        saver(img, filename=filename)

    def tensorboard_save(self, img, figure_name, step):
        img = img.squeeze(0)
        img = torch.clamp(img, min=-1, max=1)
        img = (img + 1) * 127.5
        img = img.permute(1, 0, 2, 3)
        img = img.type(torch.uint8)
        grid = make_grid(img)
        self.logger.experiment.add_image(f"{figure_name}", grid, step)

    def cossim(self, x, y):
        """
        Cosine similarity between x and y.
        """
        x = x.view(x.size(0), -1)
        y = y.view(y.size(0), -1)
        import torch.nn.functional as f
        return f.cosine_similarity(x, y, dim=1)

    def test_step(self, batch, batch_idx):
        # meta_data = batch["image"].meta
        # # meta_data=None
        # if 1 in self.cond_nums:
        #     cond1_meta_data = batch["cond1"].meta
        # if 2 in self.cond_nums:
        #     cond2_meta_data = batch["cond2"].meta
        # if 3 in self.cond_nums:
        #     cond3_meta_data = batch["cond3"].meta
        image = batch["image"]
        filename = batch["filename"]
        filename = filename[0]

        _, c, x, x_rec,cond1,cond2,cond3 = self.get_input(
            batch, self.first_stage_key, return_first_stage_outputs=True, return_original_cond=True
        )
        # self.tensorboard_save(x_rec, "rec", 0)    
        self.img_saver(x, "origin_x", filename=str(self.root_path)+'/'+str(filename)+"_origin_x")
        self.img_saver(x_rec, "ae_rec", filename=str(self.root_path)+'/'+str(filename)+"_ae_rec")
        if 1 in self.cond_nums:
            self.img_saver(cond1, "xray_0", i_type=".jpg", filename=str(self.root_path)+'/'+str(filename)+"_xray_0")
        if 2 in self.cond_nums:
            self.img_saver(cond2, "xray_1", i_type=".jpg", filename=str(self.root_path)+'/'+str(filename)+"_xray_1")
        if 3 in self.cond_nums:
            self.img_saver(cond3, "xray_2", i_type=".jpg", filename=str(self.root_path)+'/'+str(filename)+"_xray_2")
            

        # ddim_sampler = DDIMSampler(self)
        shape = (self.batch_size, self.channels, self.image_size, self.image_size, self.image_size)
        # cond_z = self.p_sample_loop(cond=c, shape=shape)
        # cond_z, _ = ddim_sampler.sample(50, batch_size=1, shape=shape, conditioning=c, verbose=False)
        # shape = (self.batch_size, self.channels, self.image_size, self.image_size, self.image_size)
        assert self.dpm_type in ["ddim", "dpm-solver"], "dpm_type must be either 'ddim' or 'dpm-solver'"
        if self.dpm_type == "ddim":
            ddim_sampler = DDIMSampler(self)
            cond_z, _ = ddim_sampler.sample(50, batch_size=self.batch_size, shape=shape, conditioning=c, verbose=False)
            print(f"ddim shape :{shape}")
        elif self.dpm_type == "dpm-solver":
            dpm_sampler = DPMSolverSampler(self)
            print(f"dpm shape :{shape}") 
            cond_z, _ = dpm_sampler.sample(50, batch_size=self.batch_size, shape=shape, conditioning=c, verbose=False)
        z = self.encode_first_stage(x).sample()

        # cos_c = self.cossim(cond_z, c).item()
        # cos_z = self.cossim(cond_z, z).item()

        # print(f"cos cond_z with cond: {cos_c}")
        # print(f"cos cond_z with ae_z: {cos_z}")

        reconstructions = self.decode_first_stage(cond_z)
        reconstructions = (reconstructions * 2 + image) / 3 

        from scripts.metrics import Structural_Similarity
        _, _, _, ssim_avg = Structural_Similarity(image, reconstructions, PIXEL_MAX=2500)
        print(f"ssim: {ssim_avg}")
        # import nibabel as nib

        # nib_rec = nib.Nifti1Image(reconstructions[0, 0, ...].unsqueeze(-1).cpu().numpy(), np.eye(4))
        # nib.save(nib_rec, os.path.join(self.root_path, "nib_rec.nii.gz"))
        self.img_saver(reconstructions, "rec", filename=str(self.root_path)+'/'+str(filename)+"_rec")

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.cond_stage_trainable:
            print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
            params = params + list(self.cond_stage_model.parameters())
        if self.learn_logvar:
            print("Diffusion model optimizing logvar")
            params.append(self.logvar)
        opt = torch.optim.AdamW(params, lr=lr)
        if self.use_scheduler:
            # scheduler = LambdaLinearScheduler(**self.scheduler_config)
            # from ldm.lr_scheduler import LambdaWarmUpCosineScheduler2
            # scheduler = LambdaWarmUpCosineScheduler2(**self.scheduler_config)

            # print("Setting up LambdaLR scheduler...")
            # scheduler = [{"scheduler": LambdaLR(opt, lr_lambda=scheduler.schedule), "interval": "step", "frequency": 1}]
            
            # print("Setting up ReduceLROnPlateau scheduler...")
            # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            #     opt,
            #     mode='min',           # 监控指标是越小越好
            #     factor=0.8,          # 学习率调整因子
            #     patience=5,         # 容忍多少个epoch指标没改善
            #     min_lr=1e-8          # 最小学习率
            # )
            # scheduler = [{"scheduler": scheduler, "interval": "epoch", "frequency": 1}]
            # scheduler = [{"scheduler": scheduler, "monitor": "train/loss_simple_epoch"}]
            from torch.optim.lr_scheduler import StepLR
            scheduler = StepLR(opt, step_size=self.scheduler_config["step_size"], gamma=self.scheduler_config["gamma"])
            scheduler = {"scheduler": scheduler, "interval": self.scheduler_config["interval"], "frequency": 1}
            return [opt], scheduler
        return opt


class DiffusionWrapper(pl.LightningModule):
    def __init__(self, diff_model_config, conditioning_key):
        super().__init__()
        # self.diffusion_model = instantiate_from_config(diff_model_config)
        self.diffusion_model = UNetModel(**diff_model_config)
        self.conditioning_key = conditioning_key
        assert self.conditioning_key in [None, "concat", "crossattn", "hybrid", "adm"]

    def forward(self, x, t, c_concat: list = None, c_crossattn: list = None):
        if self.conditioning_key is None:
            out = self.diffusion_model(x, t)
        elif self.conditioning_key == "concat":
            # print(f"x shape :{x.shape}")
            # print(c_concat[0].shape)
            # print(f" shape :{x.shape}, {c_concat[0].shape}")
            # print(f"x shape :{x.shape}")
            # print(f"c_concat shape :{c_concat[0].shape}")
            xc = torch.cat([x] + c_concat, dim=1)

            out = self.diffusion_model(xc, t)
        elif self.conditioning_key == "crossattn":
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(x, t, context=cc)
        elif self.conditioning_key == "hybrid":
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc)
        elif self.conditioning_key == "adm":
            cc = c_crossattn[0]
            out = self.diffusion_model(x, t, y=cc)
        else:
            raise NotImplementedError()

        return out


class Layout2ImgDiffusion(LatentDiffusion):
    # TODO: move all layout-specific hacks to this class
    def __init__(self, cond_stage_key, *args, **kwargs):
        assert cond_stage_key == "coordinates_bbox", 'Layout2ImgDiffusion only for cond_stage_key="coordinates_bbox"'
        super().__init__(cond_stage_key=cond_stage_key, *args, **kwargs)

    def log_images(self, batch, N=8, *args, **kwargs):
        logs = super().log_images(batch=batch, N=N, *args, **kwargs)

        key = "train" if self.training else "validation"
        dset = self.trainer.datamodule.datasets[key]
        mapper = dset.conditional_builders[self.cond_stage_key]

        bbox_imgs = []
        map_fn = lambda catno: dset.get_textual_label(dset.get_category_id(catno))
        for tknzd_bbox in batch[self.cond_stage_key][:N]:
            bboximg = mapper.plot(tknzd_bbox.detach().cpu(), map_fn, (256, 256))
            bbox_imgs.append(bboximg)

        cond_img = torch.stack(bbox_imgs, dim=0)
        logs["bbox_image"] = cond_img
        return logs