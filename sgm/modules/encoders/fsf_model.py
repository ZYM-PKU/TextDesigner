import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import lpips

from einops import rearrange
from ..diffusionmodules.model import Encoder as ResnetEncoder
from ..diffusionmodules.model import Decoder as ResnetDecoder
from ..attention import CrossAttention
from torchvision.utils import save_image

    

class ProjectionDiscriminator(nn.Module):
    """ Multi-task discriminator """
    def __init__(self, ch):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(ch, ch),
            nn.LayerNorm(ch),
            nn.SiLU(),
            nn.Linear(ch, ch),
            nn.LayerNorm(ch),
            nn.SiLU(),
            nn.Linear(ch, 1)
        )

    def forward(self, x):
        
        out = self.fc(x[...,0,0]) # (b, 1)

        return out
    

class Discriminator(nn.Module):

    def __init__(self, in_channels, ch, ch_mult, num_res_blocks, resolution, z_channels):
        super().__init__()

        self.encoder = ResnetEncoder(
            in_channels=in_channels,
            ch=ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            resolution=resolution,
            z_channels=z_channels,
            double_z=False
        )

        self.gap = nn.Sequential(
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        self.projD = ProjectionDiscriminator(z_channels)
    
    def forward(self, x):

        feat = self.encoder(x) # (b, 3, h, w) -> (b, 512, h/8, w/8)
        feat = self.gap(feat) # (b, 512, h/8, w/8) -> (b, 512, 1, 1)
        ret = self.projD(feat)

        return ret


class FSFModel(pl.LightningModule):

    def __init__(
            self,
            ckpt_path=None,
            pretraining=False,
            lr=1e-4,
            gd_w=0.1,
            w_clip=0.01,
            emb_dim=128,
            n_ref_style=6,
            n_trans_layers=8,
            ca_n_heads=8,
            ca_dim_head=64,
            c_encoder_params=None,
            s_encoder_params=None,
            decoder_params=None,
            disc_params=None
        ):
        super().__init__()

        self.emb_dim = emb_dim
        self.n_ref_style = n_ref_style
        self.w_clip = w_clip
        
        self.content_encoder = ResnetEncoder(in_channels=1, z_channels=emb_dim, double_z=False, **c_encoder_params)
        self.style_encoder = ResnetEncoder(in_channels=3, z_channels=emb_dim, double_z=False, **s_encoder_params)
        self.cross_attn = CrossAttention(query_dim=self.emb_dim, context_dim=self.emb_dim, heads=ca_n_heads, dim_head=ca_dim_head)
        transformer_block = nn.TransformerEncoderLayer(d_model=self.emb_dim, nhead=ca_n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(transformer_block, num_layers=n_trans_layers)
        self.decoder = ResnetDecoder(out_ch=3, z_channels=emb_dim*2, **decoder_params)

        if pretraining:
            self.lpips_loss = lpips.LPIPS(net='vgg', eval_mode=False)
            self.lr = lr
            self.gd_w = gd_w
            self.discriminator = Discriminator(in_channels=3, **disc_params)

            # Important: This property activates manual optimization.
            self.automatic_optimization = False
            self.init_weight()
        
        if ckpt_path is not None:
            sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
            missing, unexpected = self.load_state_dict(sd, strict=False)
            print(f"FSFModel restored from {ckpt_path} with {len(missing)} missing and {len(unexpected)} unexpected keys")

    def init_weight(self):

        def w_init(m):
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.apply(w_init)
        
    def freeze(self):

        self.eval()
        for param in self.parameters():
            param.requires_grad_(False)

    def configure_optimizers(self):

        g_params = list(self.content_encoder.parameters()) \
                + list(self.style_encoder.parameters()) \
                + list(self.cross_attn.parameters()) \
                + list(self.transformer.parameters()) \
                + list(self.decoder.parameters()) 
        
        d_params = list(self.discriminator.parameters())

        g_opt = torch.optim.AdamW(params=g_params, lr=self.lr)
        d_opt = torch.optim.AdamW(params=d_params, lr=self.lr*2)

        lr_lambda = lambda epoch: 0.95**epoch
        g_scheduler = torch.optim.lr_scheduler.LambdaLR(g_opt, lr_lambda=lr_lambda)
        d_scheduler = torch.optim.lr_scheduler.LambdaLR(d_opt, lr_lambda=lr_lambda)

        return [g_opt, d_opt], [g_scheduler, d_scheduler]

    def encode(self, c_ref, s_ref):

        b, _, h, w = c_ref.shape
        content_feat = self.content_encoder(c_ref) # (b, 1, h, w) -> (b, c, h/4, w/4)
        content_feat_cp = content_feat.clone()

        s_ref = s_ref[:, :self.n_ref_style, ...]
        s_ref = rearrange(s_ref, 'b n c h w -> (b n) c h w').contiguous() # (b, n, 3, h, w) -> (bn, 3, h, w)
        style_feat = self.style_encoder(s_ref) # (bn, 3, h, w) -> (bn, c, h/4, w/4)
        
        # cross_attn, q: content, k & v: style
        content_feat = rearrange(content_feat, 'b c h w -> b (h w) c').contiguous() # (b, hw/16, c)
        style_feat = rearrange(style_feat, 'b c h w -> b (h w) c').contiguous() # (bn, hw/16, c)
        style_feat = style_feat.reshape(b, -1, self.emb_dim) # (bn, hw/16, c) -> (b, nhw/16, c)
        out = self.cross_attn(content_feat, context=style_feat) # (b, hw/16, c)

        out = self.transformer(out) # (b, hw/16, c) -> (b, hw/16, c)

        nh = nw = int(out.shape[1]**0.5)
        out = out.permute(0,2,1).reshape(b, self.emb_dim, nh, nw).contiguous() # (b, hw/4, c) -> (b, c, h/4, w/4)
        out = torch.cat([out, content_feat_cp], dim=1) # (b, 2c, h/4, w/4)

        return out
    
    def forward(self, c_ref, s_ref):

        z = self.encode(c_ref, s_ref)
        res = self.decoder(z) # (b, 2c, h/4, w/4) -> (b, 3, h, w)

        return res

    def get_d_loss(self, real_score, fake_score):

        d_loss = F.relu(1. - real_score).mean() + F.relu(1. + fake_score).mean()
        
        return d_loss

    def get_g_loss(self, fake_score, fake, real):

        g_adv_loss = -fake_score.mean()
        g_l1_loss = F.l1_loss(fake, real, reduction="mean")
        g_lpips_loss = self.lpips_loss(fake, real).mean()

        return g_adv_loss, g_l1_loss, g_lpips_loss

    def training_step(self, batch, batch_idx):

        # load data
        c_ref = batch["c_ref"]
        s_ref = batch["s_ref"]
        real = batch["gt"]

        g_opt, d_opt = self.optimizers()

        # discriminator
        fake = self(c_ref, s_ref)
        real_score = self.discriminator(real)
        fake_score = self.discriminator(fake.detach())
        d_loss = self.get_d_loss(real_score, fake_score)
        d_loss = d_loss * self.gd_w

        d_opt.zero_grad()
        self.manual_backward(d_loss)
        d_opt.step()

        # WGAN
        # for param in self.discriminator.parameters():
        #     param.data.clamp_(-self.w_clip, self.w_clip)

        # generator
        fake_score = self.discriminator(fake)
        g_adv_loss, g_l1_loss, g_lpips_loss = self.get_g_loss(fake_score, fake, real)
        g_loss = g_adv_loss * self.gd_w + g_l1_loss + g_lpips_loss

        g_opt.zero_grad()
        self.manual_backward(g_loss)
        g_opt.step()

        self.log_dict({"g_loss/full_loss": g_loss,
                        "g_loss/adv_loss": g_adv_loss,
                        "g_loss/l1_loss": g_l1_loss,
                        "g_loss/lpips_loss": g_lpips_loss,
                        "d_loss": d_loss}, prog_bar=True)
        
        lr = g_opt.param_groups[0]["lr"]
        self.log(
            "lr_abs", lr, prog_bar=True, logger=True, on_step=False, on_epoch=True
        )

    def on_train_epoch_end(self) -> None:

        g_sche, d_sche = self.lr_schedulers()
        g_sche.step()
        d_sche.step()