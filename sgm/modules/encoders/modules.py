from contextlib import nullcontext
from functools import partial
from typing import Dict, List, Optional, Tuple, Union

import copy
import kornia
import numpy as np
import open_clip
import torch
import torch.nn as nn
from torchvision.models import vgg19
from einops import rearrange, repeat
from omegaconf import ListConfig
from random import choices

from ...modules.diffusionmodules.openaimodel import Timestep
from ...modules.encoders.swin_transformer import SwinTransformer
from ...modules.diffusionmodules.model import Encoder as ResnetEncoder
from ...modules.attention import MemoryEfficientCrossAttention
from ...util import (
    append_dims,
    autocast,
    count_params,
    default,
    disabled_train,
    instantiate_from_config
)


from torchvision import transforms
from safetensors.torch import load_file as load_safetensors
from torchvision.utils import save_image
from .fsf_model import FSFModel

# disable warning
from transformers import logging
logging.set_verbosity_error()


class AbstractEmbModel(nn.Module):

    def __init__(self):
        super().__init__()

    def freeze(self):
        return


class GeneralConditioner(nn.Module):
    
    def __init__(self, ucg_keys: List, emb_models: Union[List, ListConfig]):
        super().__init__()

        self.ucg_keys = ucg_keys
        assert len(self.ucg_keys) <= 2

        embedders = []
        for n, embconfig in enumerate(emb_models):
            embedder = instantiate_from_config(embconfig)
            assert isinstance(embedder, AbstractEmbModel)

            embedder.is_trainable = embconfig.get("is_trainable", False)
            embedder.ucg_rate = embconfig.get("ucg_rate", 0.0)
            embedder.init_model = embconfig.get("init_model", False)
            embedder.input_keys = embconfig.get("input_keys", None)
            embedder.emb_key = embconfig.get("emb_key", None)

            assert (embedder.emb_key is not None) and (embedder.input_keys is not None)

            if not embedder.is_trainable:
                embedder.train = disabled_train
                embedder.freeze()
            else:
                embedder.train()

            print(
                f"Initialized embedder #{n}: {embedder.__class__.__name__} "
                f"with {count_params(embedder, False)} params. Trainable: {embedder.is_trainable}"
            )

            embedders.append(embedder)

        self.embedders = nn.ModuleList(embedders)

    def init_model_from_obj(self, model):

        for embedder in self.embedders:
            if embedder.init_model:
                embedder.init_model_from_obj(model)

    def possibly_get_ucg_val(self, embedder: AbstractEmbModel, batch: Dict) -> Dict:

        p = embedder.ucg_rate
        b = len(batch[embedder.input_keys[0]])

        bmask = torch.bernoulli(torch.tensor([1-p]*b))

        for input_key in embedder.input_keys:
            mask = append_dims(bmask, batch[input_key].ndim).to(batch[input_key])
            batch[input_key] = batch[input_key] * mask

        return batch

    def forward(self, batch: Dict) -> Dict:

        output = dict()
        for embedder in self.embedders:
            embedding_context = nullcontext if embedder.is_trainable else torch.no_grad
            with embedding_context():
                if embedder.ucg_rate > 0.0: 
                    batch = self.possibly_get_ucg_val(embedder, batch)
                emb_out = embedder(*[batch[k] for k in embedder.input_keys])
            if not isinstance(emb_out, (list, tuple)):
                emb_out = [emb_out]

            for emb in emb_out:
                out_key = embedder.emb_key
                if out_key in output:
                    output[out_key] = torch.cat((output[out_key], emb), 1)
                else:
                    output[out_key] = emb
        
        return output

    def get_unconditional_conditioning(self, batch: Dict) -> Dict:

        device = next(self.embedders[0].parameters()).device
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device)
        
        batch_uc = copy.deepcopy(batch)
        for ucg_key in self.ucg_keys:
            batch_uc[ucg_key] = torch.zeros_like(batch_uc[ucg_key])
            
        ucg_rates = []
        for embedder in self.embedders:
            ucg_rates.append(embedder.ucg_rate)
            embedder.ucg_rate = 0.0

        c = self(batch)
        uc = self(batch_uc)
        conds = [c, uc]

        for embedder, rate in zip(self.embedders, ucg_rates):
            embedder.ucg_rate = rate

        return conds
    

### Vector Encoders
class ConcatTimestepEmbedderND(AbstractEmbModel):
    """embeds each dimension independently and concatenates them"""

    def __init__(self, outdim):
        super().__init__()
        self.timestep = Timestep(outdim)
        self.outdim = outdim
    
    def freeze(self):
        self.eval()

    def forward(self, x):
        if x.ndim == 1:
            x = x[:, None]
        assert len(x.shape) == 2
        b, dims = x.shape[0], x.shape[1]
        x = rearrange(x, "b d -> (b d)")
        emb = self.timestep(x)
        emb = rearrange(emb, "(b d) d2 -> b (d d2)", b=b, d=dims, d2=self.outdim)
        return emb


### Image Encoders
class IdentityEncoder(AbstractEmbModel):
    def encode(self, x):
        return x
    def freeze(self):
        return
    def forward(self, x):
        return x


class SpatialRescaler(AbstractEmbModel):
    def __init__(
        self,
        method="bilinear",
        multiplier=0.5
    ):
        super().__init__()
        assert method in [
            "nearest",
            "linear",
            "bilinear",
            "trilinear",
            "bicubic",
            "area",
        ]
        self.multiplier = multiplier
        self.interpolator = partial(torch.nn.functional.interpolate, mode=method)

    def forward(self, x):
        x = self.interpolator(x, scale_factor=self.multiplier)
        return x

    def encode(self, x):
        return self(x)
    

class VGG19(AbstractEmbModel):

    def __init__(self, model_path, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = vgg19()
        self.model.load_state_dict(torch.load(model_path))
        self.model.classifier = nn.Sequential(*list(self.model.classifier.children())[:-1])

        self.transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    
    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        x = self.transform(x)
        z = self.model(x)
        z = z[:,None,:]
        return z


class LatentEncoder(AbstractEmbModel):

    def __init__(self, scale_factor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale_factor = scale_factor
        self.model = None

    def init_model_from_obj(self, model):
        self.model = model

    def forward(self, x):
        z = self.model.encode(x)
        z = self.scale_factor * z
        return z


class CLIPReferenceEmbedder(AbstractEmbModel):

    def __init__(
        self,
        arch="ViT-H-14",
        version="laion2b_s32b_b79k"
    ):
        super().__init__()
        model, _, _ = open_clip.create_model_and_transforms(
            arch,
            device=torch.device("cpu"),
            pretrained=version,
        )
        del model.transformer
        self.model = model

        self.register_buffer(
            "mean", torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False
        )
        self.register_buffer(
            "std", torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False
        )

    def freeze(self):
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
    
    def preprocess(self, x):
        # renormalize according to clip
        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x

    @autocast
    def forward(self, reference):

        b, l, h, w = reference.shape
        reference = reference.reshape(b*l, 1, h, w).tile(1,3,1,1) # bl, 3, h, w
        reference = self.preprocess(reference)

        z = self.model.visual(reference).to(reference.dtype) # bl, 1024
        z = z.reshape(b, l, 1024)

        return z

    def encode(self, reference):
        return self(reference)


class SwinReferenceEmbedder(AbstractEmbModel):

    def __init__(
        self,
        img_size=224,
        patch_size=4,
        in_chans=3,
        num_classes=8464,
        embed_dim=128,
        depths=[2,2,18,2],
        num_heads=[4,8,16,32],
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        drop_path_rate=0.5,
        ape=False,
        patch_norm=True,
        ckpt_path=None
    ):
        super().__init__()
        
        model = SwinTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            ape=ape,
            patch_norm=patch_norm
        )

        self.img_size = img_size
        self.in_chans = in_chans
        self.model = model

        if ckpt_path is not None:
            self.model.load_state_dict(torch.load(ckpt_path, map_location="cpu")["model"], strict=True)

    def freeze(self):
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    @autocast
    def forward(self, reference):

        b, l, h, w = reference.shape
        reference = reference.reshape(b*l,1,h,w).tile(1,self.in_chans,1,1) # bl, 3, h, w
        if h != self.img_size or w != self.img_size:
            reference = kornia.geometry.resize(
                reference,
                (224, 224),
                interpolation="bicubic",
                align_corners=True,
                antialias=True
            )

        z = self.model.forward_features(reference).to(reference.dtype) # bl, 1024
        z = z.reshape(b, l, 1024)

        return z

    def encode(self, reference):
        return self(reference)
    

class FSFReferenceEmbedder(AbstractEmbModel):
    def __init__(
            self,
            ckpt_path=None,
            n_ref_content=1,
            n_ref_style=6,
            in_channel=1,
            n_heads=8,
            dim_head=64,
            num_res_char=2,
            num_res_font=2,
            ch_mult_char=None,
            ch_mult_font=None,
            **encoder_kwargs
        ):
        super().__init__()

        self.emb_dim = encoder_kwargs['z_channels']
        self.n_ref_content = n_ref_content
        self.n_ref_style = n_ref_style
        self.in_channel = in_channel
        
        self.content_encoder = ResnetEncoder(in_channels=n_ref_content*in_channel, num_res_blocks=num_res_char, ch_mult=ch_mult_char, **encoder_kwargs)
        self.style_encoder = ResnetEncoder(in_channels=in_channel, num_res_blocks=num_res_font, ch_mult=ch_mult_font, **encoder_kwargs)
        self.attn = MemoryEfficientCrossAttention(query_dim=self.emb_dim, context_dim=self.emb_dim, heads=n_heads, dim_head=dim_head)

        self.to_out = nn.Sequential(
            nn.GroupNorm(num_groups=self.emb_dim//4, num_channels=self.emb_dim),
            nn.SiLU(),
            nn.Conv2d(in_channels=self.emb_dim, out_channels=self.emb_dim, kernel_size=3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d(in_channels=self.emb_dim, out_channels=self.emb_dim, kernel_size=1, padding=0),
            nn.GroupNorm(num_groups=self.emb_dim//4, num_channels=self.emb_dim),
            nn.SiLU(),
            nn.Conv2d(in_channels=self.emb_dim, out_channels=self.emb_dim, kernel_size=3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d(in_channels=self.emb_dim, out_channels=self.emb_dim, kernel_size=1, padding=0)
        )
        
        if ckpt_path is not None:
            miss, unexcept = self.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=False)
        
    def freeze(self):

        self.eval()
        for param in self.parameters():
            param.requires_grad = False
        self.to_out.train()
        for param in self.to_out.parameters():
            param.requires_grad = True

    def forward(self, c_ref, s_ref):

        b, l, h, w = c_ref.shape
        c_ref = c_ref.reshape(-1,1,64,64).tile(1,self.in_channel,1,1) # (b, l, h, w) -> (bl, 1, h, w)
        content_feat = self.content_encoder(c_ref) # (bl, c, h/2, w/2)
        
        s_ref = s_ref.reshape(-1,3,64,64) # (b, l, 3, h, w) -> (bl, 3, h, w)
        style_feat = self.style_encoder(s_ref) # (bl, c, h/2, w/2)
        
        # cross attn, q: content, k & v: style
        style_feat = rearrange(style_feat, 'b c h w -> b (h w) c').contiguous()
        content_feat = rearrange(content_feat, 'b c h w -> b (h w) c').contiguous()
        style_feat = style_feat.tile(1,self.n_ref_style,1)
        out = self.attn(content_feat, context=style_feat) # (bl, hw/4, c)

        out = out.permute(0,2,1) # (bl, c, hw/4)
        out = out.reshape(b*l, -1, h//2, w//2) # (bl, c, h/2, w/2)
        out = self.to_out(out) # (bl, c/4, h/4, w/4)
        out = out.reshape(b, l, -1) # 8192

        return out

    def _forward(self, c_ref, s_ref, ref_len):
        """
        input:
            - c_ref: b, l, h, w (l: seq_len)
            - s_ref: b, l, 3, h, w
            - ref_len: b, 1
        output:
            - out: b, l, 2048
        """

        b, l, h, w = c_ref.shape
        c_ref = c_ref.reshape(-1,1,64,64).tile(1,self.in_channel,1,1) # (b, l, h, w) -> (bl, 1, h, w)
        content_feat = self.content_encoder(c_ref) # (bl, c, h/2, w/2)

        s_ref_l = []
        for i in range(b):
            n = ref_len[i]
            s_ref_i = s_ref[i][:n] # (n, 3, h, w)
            s_ref_rand = choices(list(s_ref_i), k=n*self.n_ref_style)
            s_ref_rand = torch.cat(s_ref_rand, dim=0).reshape(n, self.n_ref_style, 3, h, w) # (n, nrs, 3, h, w)
            if l > n:
                s_ref_pad = torch.zeros(l-n, self.n_ref_style, 3, h, w).to(s_ref_rand)
                s_ref_rand = torch.cat([s_ref_rand, s_ref_pad], dim=0) # (l, nrs, 3, h, w)
            s_ref_l.append(s_ref_rand)
        s_ref = torch.stack(s_ref_l, dim=0) # (b, l, nrs, 3, h, w)

        s_ref = s_ref.reshape(-1, 3, h, w) # (blnrs, 3, h, w)
        style_feat = self.style_encoder(s_ref) # (blnrs, c, h/8, w/8)
        
        # cross attn, q: content, k & v: style
        content_feat = rearrange(content_feat, 'b c h w -> b (h w) c').contiguous() # (bl, hw/4, c)
        style_feat = rearrange(style_feat, 'b c h w -> b (h w) c').contiguous() # (blnrs, hw/64, c)
        style_feat = style_feat.reshape(b*l, -1, self.emb_dim) # (bl, nrshw/64, c)
        out = self.attn(content_feat, context=style_feat) # (bl, hw/4, c)

        out = out.permute(0,2,1) # (bl, c, hw/4)
        out = out.reshape(b*l, -1, h//2, w//2) # (bl, c, h/2, w/2)
        out = self.to_out(out) # (bl, c/4, h/4, w/4)
        out = out.reshape(b, l, -1) # 8192

        return out


class FSFReferenceEmbedder2(AbstractEmbModel):
    def __init__(
            self,
            ckpt_path=None,
            emb_dim=128,
            n_ref_style=6,
            n_trans_layers=8,
            ca_n_heads=8,
            ca_dim_head=64,
            c_encoder_params=None,
            s_encoder_params=None
        ):
        super().__init__()

        self.emb_dim = emb_dim
        self.n_ref_style = n_ref_style

        self.model = FSFModel(
            ckpt_path=ckpt_path,
            pretraining=False,
            emb_dim=emb_dim,
            n_ref_style=n_ref_style,
            n_trans_layers=n_trans_layers,
            ca_n_heads=ca_n_heads,
            ca_dim_head=ca_dim_head,
            c_encoder_params=c_encoder_params,
            s_encoder_params=s_encoder_params
        )

        self.to_out = nn.Sequential(
            nn.InstanceNorm2d(num_features=self.emb_dim*2, affine=True),
            nn.SiLU(),
            nn.Conv2d(in_channels=self.emb_dim*2, out_channels=self.emb_dim*2, kernel_size=3, padding=1, stride=2),
            nn.InstanceNorm2d(num_features=self.emb_dim*2, affine=True),
            nn.SiLU(),
            nn.Conv2d(in_channels=self.emb_dim*2, out_channels=self.emb_dim, kernel_size=1, padding=0),
            nn.InstanceNorm2d(num_features=self.emb_dim, affine=True),
            nn.SiLU(),
            nn.Conv2d(in_channels=self.emb_dim, out_channels=self.emb_dim//2, kernel_size=1, padding=0)
        )
        
    def freeze(self):

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.to_out.train()
        for param in self.to_out.parameters():
            param.requires_grad = True

    def forward(self, c_ref, s_ref, ref_len):
        """
        input:
            - c_ref: b, l, h, w (l: seq_len)
            - s_ref: b, l, 3, h, w
            - ref_len: b, 1
        output:
            - out: b, l, 2048
        """

        b, l, h, w = c_ref.shape

        s_ref_l = []
        for i in range(b):
            n = ref_len[i]
            if n > 0:
                s_ref_i = s_ref[i][:n] # (n, 3, h, w)
                s_ref_rand = choices(list(s_ref_i), k=n*self.n_ref_style)
                s_ref_rand = torch.cat(s_ref_rand, dim=0).reshape(n, self.n_ref_style, 3, h, w) # (n, nrs, 3, h, w)
                if l > n:
                    s_ref_pad = torch.zeros(l-n, self.n_ref_style, 3, h, w).to(s_ref_rand)
                    s_ref_rand = torch.cat([s_ref_rand, s_ref_pad], dim=0) # (l, nrs, 3, h, w)
            else:
                s_ref_rand = torch.zeros((l, self.n_ref_style, 3, h, w)).to(s_ref)
            s_ref_l.append(s_ref_rand)
        s_ref = torch.stack(s_ref_l, dim=0) # (b, l, nrs, 3, h, w)

        c_ref_b = c_ref.reshape(b*l, 1, h, w) # (bl, 1, h, w)
        s_ref_b = s_ref.reshape(b*l, self.n_ref_style, 3, h, w) # (bl, nrs, 3, h, w)

        out = self.model.encode(c_ref_b, s_ref_b) # (bl, 2c, h/4, w/4)
        out = self.to_out(out) # (bl, c/2, h/8, w/8)
        out = out.reshape(b, l, -1) # (b, l, 8192)

        return out