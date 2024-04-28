import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import lpips

from einops import rearrange
from ..diffusionmodules.model import Encoder as ResnetEncoder
from ..diffusionmodules.model import Decoder as ResnetDecoder
from ..attention import MemoryEfficientCrossAttention
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
            lpips_w=0.1,
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
        self.attn1 = MemoryEfficientCrossAttention(query_dim=self.emb_dim, context_dim=self.emb_dim, heads=ca_n_heads, dim_head=ca_dim_head)
        self.attn2 = MemoryEfficientCrossAttention(query_dim=self.emb_dim, context_dim=self.emb_dim, heads=ca_n_heads, dim_head=ca_dim_head)
        transformer_block = nn.TransformerEncoderLayer(d_model=self.emb_dim, nhead=ca_n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(transformer_block, num_layers=n_trans_layers)
        if decoder_params is not None:
            self.content_decoder = ResnetDecoder(out_ch=1, z_channels=emb_dim*2, **decoder_params)
            self.style_decoder = ResnetDecoder(out_ch=3, z_channels=emb_dim*2, **decoder_params)
            self.to_out = nn.Sequential(
                nn.Conv2d(in_channels=4, out_channels=64, kernel_size=3, padding=1),
                nn.InstanceNorm2d(num_features=64), nn.SiLU(),
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
                nn.InstanceNorm2d(num_features=128), nn.SiLU(),
                nn.Conv2d(in_channels=128, out_channels=3, kernel_size=1, padding=0)
            )

        if pretraining:
            self.lpips_loss = lpips.LPIPS(net='vgg', eval_mode=False)
            self.lr = lr
            self.gd_w = gd_w
            self.lpips_w = lpips_w
            self.content_discriminator = Discriminator(in_channels=1, **disc_params)
            self.style_discriminator = Discriminator(in_channels=3, **disc_params)

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
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.apply(w_init)
        
        for param in self.parameters():
            param.requires_grad_(True)
        for param in self.lpips_loss.parameters():
            param.requires_grad_(False)

        
    def freeze(self):

        for param in self.parameters():
            param.requires_grad_(False)

    def configure_optimizers(self):

        g_params = list(self.content_encoder.parameters()) \
                + list(self.style_encoder.parameters()) \
                + list(self.attn1.parameters()) \
                + list(self.attn2.parameters()) \
                + list(self.transformer.parameters()) \
                + list(self.content_decoder.parameters()) \
                + list(self.style_decoder.parameters()) \
                + list(self.to_out.parameters())
        
        d_params = list(self.content_discriminator.parameters()) \
                + list(self.style_discriminator.parameters())

        g_opt = torch.optim.AdamW(params=g_params, lr=self.lr)
        d_opt = torch.optim.AdamW(params=d_params, lr=self.lr*2)

        lr_lambda = lambda epoch: 0.95**epoch
        g_scheduler = torch.optim.lr_scheduler.LambdaLR(g_opt, lr_lambda=lr_lambda)
        d_scheduler = torch.optim.lr_scheduler.LambdaLR(d_opt, lr_lambda=lr_lambda)

        return [g_opt, d_opt], [g_scheduler, d_scheduler]
    
    def AdaIN(self, c_feat, s_feat):

        c_feat_mean = c_feat.mean(dim=-2, keepdims=True) # (b, 1, d)
        c_feat_std = c_feat.std(dim=-2, keepdims=True) # (b, 1, d)
        
        s_feat_std = s_feat.mean(dim=-2, keepdims=True) # (b, 1, d)
        s_feat_mean = s_feat.std(dim=-2, keepdims=True) # (b, 1, d)

        feat = (c_feat - c_feat_mean) / c_feat_std
        feat = feat * s_feat_std + s_feat_mean

        return feat
    
    def fusion(self, c_feat, s_feat):

        c_feat = self.AdaIN(c_feat, s_feat) # (b, hw/16, c)
        s_feat = torch.cat([c_feat, s_feat], dim=1) # (b, (n+1)hw/16, c)

        c_feat = self.attn1(c_feat, s_feat) # (b, hw/16, c)
        c_feat = self.transformer(c_feat) # (b, hw/16, c) -> (b, hw/16, c)
        out = self.attn2(c_feat, s_feat) # (b, hw/16, c)

        return out

    def encode(self, c_ref, s_ref):
        """
        input:
            - c_ref: b, 1, h, w
            - s_ref: b, n, 3, h, w
        output:
            - out: b, 2c, h/4, w/4
        """

        b, _, h, w = c_ref.shape

        content_feat = self.content_encoder(c_ref) # (b, 1, h, w) -> (b, c, h/8, w/8)
        content_feat_cp = content_feat.clone()
 
        s_ref = rearrange(s_ref, 'b n c h w -> (b n) c h w').contiguous() # (b, n, 3, h, w) -> (bn, 3, h, w)
        style_feat = self.style_encoder(s_ref) # (bn, 3, h, w) -> (bn, c, h/8, w/8)
        
        content_feat = rearrange(content_feat, 'b c h w -> b (h w) c').contiguous() # (b, hw/16, c)
        style_feat = rearrange(style_feat, 'b c h w -> b (h w) c').contiguous() # (bn, hw/64, c)
        style_feat = style_feat.reshape(b, -1, self.emb_dim) # (bn, hw/64, c) -> (b, nhw/64, c)
        out = self.fusion(content_feat, style_feat) # (b, hw/16, c)

        nh = nw = int(out.shape[1]**0.5)
        out = out.permute(0,2,1).reshape(b, self.emb_dim, nh, nw).contiguous() # (b, hw/4, c) -> (b, c, h/4, w/4)
        out = torch.cat([out, content_feat_cp], dim=1) # (b, 2c, h/4, w/4)

        return out
    
    def forward(self, c_ref, s_ref):

        z = self.encode(c_ref, s_ref)
        content_res = self.content_decoder(z)
        style_res = self.style_decoder(z)
        style_res = torch.cat([style_res, content_res], dim=1)
        style_res = self.to_out(style_res)

        return content_res, style_res

    def get_d_loss(self, real_score, fake_score):

        d_loss = F.relu(1. - real_score).mean() + F.relu(1. + fake_score).mean()
        
        return d_loss

    def get_g_loss(self, c_fake_score, s_fake_score, c_fake, s_fake, c_real, s_real):

        g_adv_loss = -(c_fake_score + s_fake_score).mean()
        g_l1_loss = F.l1_loss(c_fake, c_real, reduction="mean") \
                + F.l1_loss(s_fake, s_real, reduction="mean")
        g_lpips_loss = self.lpips_loss.forward(s_fake, s_real).mean()

        return g_adv_loss, g_l1_loss, g_lpips_loss

    def training_step(self, batch, batch_idx):

        # load data
        c_ref = batch["c_ref"]
        s_ref = batch["s_ref"]
        c_gt = batch["c_gt"]
        s_gt = batch["s_gt"]
        
        g_opt, d_opt = self.optimizers()

        # discriminator
        c_fake, s_fake = self(c_ref, s_ref)
        c_real_score = self.content_discriminator(c_gt)
        c_fake_score = self.content_discriminator(c_fake.detach())
        c_d_loss = self.get_d_loss(c_real_score, c_fake_score)
        
        s_real_score = self.style_discriminator(s_gt)
        s_fake_score = self.style_discriminator(s_fake.detach())
        s_d_loss = self.get_d_loss(s_real_score, s_fake_score)
        
        d_loss = (c_d_loss + s_d_loss) * self.gd_w

        d_opt.zero_grad()
        self.manual_backward(d_loss)
        d_opt.step()

        # WGAN
        # for param in self.discriminator.parameters():
        #     param.data.clamp_(-self.w_clip, self.w_clip)

        # generator
        c_fake_score = self.content_discriminator(c_fake)
        s_fake_score = self.style_discriminator(s_fake)
        g_adv_loss, g_l1_loss, g_lpips_loss = self.get_g_loss(c_fake_score, s_fake_score, c_fake, s_fake, c_gt, s_gt)
        g_loss = g_adv_loss * self.gd_w + g_l1_loss + g_lpips_loss * self.lpips_w

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