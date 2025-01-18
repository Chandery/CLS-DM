import torch
import torch.nn.functional as F
import lightning as pl
import torchvision

from ..autoencoderkl.model import Encoder, Decoder
from ..autoencoderkl.quantize import VectorQuantizer2 as VectorQuantizer, GumbelQuantize
from .vqlosses import VQLPIPSWithDiscriminator, DummyLoss


class VQModel(pl.LightningModule):
    def __init__(
        self,
        ddconfig,
        lossconfig,
        n_embed,
        embed_dim,
        base_learning_rate,
        ckpt_path=None,
        ignore_keys=[],
        image_key="image",
        colorize_nlabels=None,
        monitor=None,
        remap=None,
        sane_index_shape=False,  # tell vector quantizer to return indices as bhw
        root_path=None,
    ):
        super(VQModel, self).__init__()
        # * manual optimization
        self.automatic_optimization = False

        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        # self.loss = instantiate_from_config(lossconfig)
        if lossconfig is None:
            self.loss = DummyLoss()
        else:
            self.loss = VQLPIPSWithDiscriminator(**lossconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25, remap=remap, sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv3d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv3d(embed_dim, ddconfig["z_channels"], 1)
        self.learning_rate = base_learning_rate
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels) == int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

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

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float()

    def training_step(self, batch, batch_idx):
        # x = self.get_input(batch, self.image_key)
        opt_ae, opt_disc = self.optimizers()
        x = batch
        xrec, qloss = self(x)

        # if optimizer_idx == 0:
        # autoencode
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step, last_layer=self.get_last_layer(), split="train")

        opt_ae.zero_grad()
        self.manual_backward(aeloss)
        opt_ae.step()

        self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        # discriminator
        discloss, log_dict_disc = self.loss(
            qloss, x, xrec, 1, self.global_step, last_layer=self.get_last_layer(), split="train"
        )
        opt_disc.zero_grad()
        self.manual_backward(discloss)
        opt_disc.step()

        self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        if self.current_epoch % 10 == 0:
            inputs = batch
            reconstructions, qloss = self(inputs)
            aeloss, log_dict_ae = self.loss(
                qloss, inputs, reconstructions, 0, self.global_step, last_layer=self.get_last_layer(), split="val"
            )

            discloss, log_dict_disc = self.loss(
                qloss, inputs, reconstructions, 1, self.global_step, last_layer=self.get_last_layer(), split="val"
            )
            # rec_loss = log_dict_ae["val/rec_loss"]
            # self.log("val/rec_loss", lo, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
            # self.log("val/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
            self.log_dict(log_dict_ae)
            self.log_dict(log_dict_disc)

            inputs = self.to_image(inputs)
            reconstructions = self.to_image(reconstructions)

            grid = torchvision.utils.make_grid(reconstructions)
            self.logger.experiment.add_image("val_images", grid, self.global_step)

            grid = torchvision.utils.make_grid(inputs)
            self.logger.experiment.add_image("val_inputs", grid, self.global_step)

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.quantize.parameters())
            + list(self.quant_conv.parameters())
            + list(self.post_quant_conv.parameters()),
            lr=lr,
            betas=(0.5, 0.9),
        )
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(), lr=lr, betas=(0.5, 0.9))
        return opt_ae, opt_disc

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def to_image(self, x):
        x = torch.clamp(x, min=-1, max=1)
        x = (x + 1) * 127.5
        x = x.squeeze(0).permute(1, 0, 2, 3)
        x = x.type(torch.uint8)
        # x = x.cpu().numpy()
        return x
