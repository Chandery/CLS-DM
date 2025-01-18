import os
import math
import torch
import torch.nn.functional as F

import lightning as pl
import SimpleITK as sitk


# from main import instantiate_from_config
# from taming.modules.util import SOSProvider
from .util import Identity
from .mingpt import GPT, sample_with_past
from .vqgan import VQModel


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class Net2NetTransformer(pl.LightningModule):
    def __init__(
        self,
        transformer_config,
        first_stage_config,
        cond_stage_config,
        root_path,
        base_learning_rate=1e-5,
        permuter_config=None,
        ckpt_path=None,
        ignore_keys=[],
        first_stage_key="image",
        cond_stage_key="depth",
        downsample_cond_size=-1,
        pkeep=1.0,
        sos_token=0,
        unconditional=False,
    ):
        super().__init__()
        self.be_unconditional = unconditional
        self.sos_token = sos_token
        self.first_stage_key = first_stage_key
        self.cond_stage_key = cond_stage_key
        self.learning_rate = base_learning_rate
        self.root_path = root_path
        self.init_first_stage_from_ckpt(first_stage_config)
        self.init_cond_stage_from_ckpt(cond_stage_config)
        if permuter_config is None:
            permuter_config = {"target": "taming.modules.transformer.permuter.Identity"}
        # self.permuter = instantiate_from_config(config=permuter_config)
        self.permuter = Identity()
        # self.transformer = instantiate_from_config(config=transformer_config)
        self.transformer = GPT(**transformer_config)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.downsample_cond_size = downsample_cond_size
        self.pkeep = pkeep

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        for k in sd.keys():
            for ik in ignore_keys:
                if k.startswith(ik):
                    self.print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def init_first_stage_from_ckpt(self, config):
        # model = instantiate_from_config(config)
        model = VQModel(**config)
        model.eval()
        self.first_stage_model = model
        self.first_stage_model = self.first_stage_model.eval()
        self.first_stage_model.train = disabled_train

    def init_cond_stage_from_ckpt(self, config):
        # if config == "__is_first_stage__":
        #     print("Using first stage also as cond stage.")
        # * for us comparative experiments  fisrt_stage_model == cond_stage_model
        self.cond_stage_model = self.first_stage_model
        self.cond_stage_model.eval()
        # elif config == "__is_unconditional__" or self.be_unconditional:
        #     print(
        #         f"Using no cond stage. Assuming the training is intended to be unconditional. "
        #         f"Prepending {self.sos_token} as a sos token."
        #     )
        #     self.be_unconditional = True
        #     self.cond_stage_key = self.first_stage_key
        #     self.cond_stage_model = SOSProvider(self.sos_token)
        # else:
        #     model = instantiate_from_config(config)
        #     model = model.eval()
        #     model.train = disabled_train
        #     self.cond_stage_model = model

    def forward(self, x, c):
        # one step to produce the logits
        quant_z, z_indices = self.encode_to_z(x)  # * [1,4,16,8,8]
        print("quantz shape", quant_z.shape)
        # _, c_indices = self.encode_to_c(c)
        _, c_indices = self.encode_to_c(c)

        if self.training and self.pkeep < 1.0:
            mask = torch.bernoulli(self.pkeep * torch.ones(z_indices.shape, device=z_indices.device))
            mask = mask.round().to(dtype=torch.int64)
            r_indices = torch.randint_like(z_indices, self.transformer.config.vocab_size)
            a_indices = mask * z_indices + (1 - mask) * r_indices
        else:
            a_indices = z_indices

        cz_indices = torch.cat((c_indices, a_indices), dim=1)
        print(cz_indices.shape)  # [1,2048]

        # target includes all sequence elements (no need to handle first one
        # differently because we are conditioning)
        target = z_indices
        # make the prediction
        logits, _ = self.transformer(cz_indices[:, :-1])
        # cut off conditioning outputs - output i corresponds to p(z_i | z_{<i}, c)
        logits = logits[:, c_indices.shape[1] - 1 :]

        return logits, target

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float("Inf")
        return out

    @torch.no_grad()
    def sample(self, x, c, steps, temperature=1.0, sample=False, top_k=None, callback=lambda k: None):
        x = torch.cat((c, x), dim=1)
        block_size = self.transformer.get_block_size()
        assert not self.transformer.training
        if self.pkeep <= 0.0:
            # one pass suffices since input is pure noise anyway
            assert len(x.shape) == 2
            noise_shape = (x.shape[0], steps - 1)
            # noise = torch.randint(self.transformer.config.vocab_size, noise_shape).to(x)
            noise = c.clone()[:, x.shape[1] - c.shape[1] : -1]
            x = torch.cat((x, noise), dim=1)
            logits, _ = self.transformer(x)
            # take all logits for now and scale by temp
            logits = logits / temperature
            # optionally crop probabilities to only the top k options
            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)
            # apply softmax to convert to probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution or take the most likely
            if sample:
                shape = probs.shape
                probs = probs.reshape(shape[0] * shape[1], shape[2])
                ix = torch.multinomial(probs, num_samples=1)
                probs = probs.reshape(shape[0], shape[1], shape[2])
                ix = ix.reshape(shape[0], shape[1])
            else:
                _, ix = torch.topk(probs, k=1, dim=-1)
            # cut off conditioning
            x = ix[:, c.shape[1] - 1 :]
        else:
            for k in range(steps):
                callback(k)
                print("x size:", x.size(1))
                assert x.size(1) <= block_size  # make sure model can see conditioning
                x_cond = x if x.size(1) <= block_size else x[:, -block_size:]  # crop context if needed
                logits, _ = self.transformer(x_cond)
                # pluck the logits at the final step and scale by temperature
                logits = logits[:, -1, :] / temperature
                # optionally crop probabilities to only the top k options
                if top_k is not None:
                    logits = self.top_k_logits(logits, top_k)
                # apply softmax to convert to probabilities
                probs = F.softmax(logits, dim=-1)
                # sample from the distribution or take the most likely
                if sample:
                    ix = torch.multinomial(probs, num_samples=1)
                else:
                    _, ix = torch.topk(probs, k=1, dim=-1)
                # append to the sequence and continue
                x = torch.cat((x, ix), dim=1)
            # cut off conditioning
            x = x[:, c.shape[1] :]
        return x

    @torch.no_grad()
    def encode_to_z(self, x):
        quant_z, _, info = self.first_stage_model.encode(x)
        indices = info[2].view(quant_z.shape[0], -1)
        indices = self.permuter(indices)
        return quant_z, indices

    @torch.no_grad()
    def encode_to_c(self, c):
        if self.downsample_cond_size > -1:
            c = F.interpolate(c, size=(self.downsample_cond_size, self.downsample_cond_size))
        quant_c, _, [_, _, indices] = self.cond_stage_model.encode(c)
        # if len(indices.shape) > 2:
        indices = indices.view(c.shape[0], -1)
        return quant_c, indices

    @torch.no_grad()
    def decode_to_img(self, index, zshape):
        index = self.permuter(index, reverse=True).long()
        bhwc = (zshape[0], zshape[2], zshape[3], zshape[4], zshape[1])
        quant_z = self.first_stage_model.quantize.get_codebook_entry(index.reshape(-1), shape=bhwc)
        print("decode to img quant_Z shape:", quant_z.shape)
        x = self.first_stage_model.decode(quant_z)
        return x

    def get_input(self, key, batch):
        x = batch[key]
        if len(x.shape) == 3:
            x = x[..., None]
        if len(x.shape) == 4:
            x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        if x.dtype == torch.double:
            x = x.float()
        return x

    def get_xc(self, batch, N=None):
        x = self.get_input(self.first_stage_key, batch)
        c = self.get_input(self.cond_stage_key, batch)
        if N is not None:
            x = x[:N]
            c = c[:N]
        return x, c

    def shared_step(self, batch, batch_idx):
        # x, c = self.get_xc(batch)
        x, cond_dict = batch
        c = cond_dict["cond"]
        logits, target = self(x, c)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log("train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log("val/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        def to_image(x):
            x = torch.clamp(x, min=-1, max=1)
            x = (x + 1) * 127.5
            # x = x.squeeze(0).permute(1, 0, 2, 3)
            x = x.type(torch.uint8)
            x = x.cpu().numpy()
            return x

        x, cond_dict = batch
        c = cond_dict["cond"]

        aug = to_image(x)
        image = sitk.GetImageFromArray(aug)
        sitk.WriteImage(image, os.path.join(self.root_path, f"origin_{batch_idx}.mhd"))

        cond_image = to_image(c)
        image = sitk.GetImageFromArray(cond_image)
        sitk.WriteImage(image, os.path.join(self.root_path, f"before_augmentation{batch_idx}.mhd"))

        quant_c, c_indices = self.encode_to_z(c)
        shape = quant_c.shape
        print(f"shape:{shape}")

        dummy_indices = torch.zeros_like(c_indices)
        # idx = self.sample(x=dummy_indices, c=c_indices, steps=4)
        idx = sample_with_past(c_indices, model=self.transformer, steps=1024)
        print("sample_with_past idx shape:", idx.shape)
        # cz_indices = torch.cat((c_indices, dummy_indices), dim=1)
        # x_indices = self.sample(x, cz_indices, steps=x.shape[1], sample=True, top_k=1)
        # logits, _ = self.transformer(cz_indices[:, :-1])
        # logits = logits[:, c_indices.shape[1] - 1 :]
        # logits = logits.reshape(shape[0], shape[2], shape[3], shape[4], -1)
        # print(logits.shape)
        # probs = torch.nn.functional.softmax(logits, dim=-1)
        # _, ix = torch.topk(probs, k=1, dim=-1)
        # print(ix.shape)

        reconstructions = self.decode_to_img(idx, shape)
        # print(reconstructions.shape)
        reconstructions = to_image(reconstructions)
        image = sitk.GetImageFromArray(reconstructions)
        sitk.WriteImage(image, os.path.join(self.root_path, f"reconstructions_{batch_idx}.mhd"))

    def configure_optimizers(self):
        """
        Following minGPT:
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.transformer.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("pos_emb")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.transformer.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(param_dict.keys() - union_params) == 0, (
            "parameters %s were not separated into either decay/no_decay set!" % (str(param_dict.keys() - union_params),)
        )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=self.learning_rate, betas=(0.9, 0.95))
        return optimizer
