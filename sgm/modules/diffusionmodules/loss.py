from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import ListConfig
from torchvision.utils import save_image
from torchvision.transforms import GaussianBlur
from ...util import append_dims, instantiate_from_config
from lpips import LPIPS


class StandardDiffusionLoss(nn.Module):

    def __init__(
        self,
        type="l2",
        offset_noise_level=0.0,
        *args, **kwarg
    ):
        super().__init__()

        assert type in ["l2", "l1"]

        self.type = type
        self.offset_noise_level = offset_noise_level

    def __call__(self, input, model_output, w, *args, **kwargs):

        loss = self.get_diff_loss(model_output, input, w)
        loss = loss.mean()
        loss_dict = {"loss": loss}

        return loss, loss_dict

    def get_diff_loss(self, model_output, target, w):

        if self.type == "l2":
            return torch.mean(
                (w * (model_output - target) ** 2).reshape(target.shape[0], -1), 1
            )
        elif self.type == "l1":
            return torch.mean(
                (w * (model_output - target).abs()).reshape(target.shape[0], -1), 1
            )
        

class FullLoss(StandardDiffusionLoss):

    def __init__(
        self,
        g_kernel_size=3,
        g_sigma=0.5,
        min_attn_size=16,
        lambda_local_loss=None,
        lambda_style_loss=None,
        lambda_ocr_loss=None,
        predictor_config=None,
        *args, **kwarg
    ):
        super().__init__(*args, **kwarg)

        self.min_attn_size = min_attn_size
        self.lambda_local_loss = lambda_local_loss
        self.lambda_ocr_loss = lambda_ocr_loss
        self.lambda_style_loss = lambda_style_loss

        self.lpips_loss = LPIPS(net="vgg", eval_mode=False)
        self.gaussian_blur = GaussianBlur(kernel_size=g_kernel_size, sigma=g_sigma)

        if lambda_ocr_loss is not None:
            self.predictor = instantiate_from_config(predictor_config)

    def __call__(self, input, model_output, w, batch, model_output_decoded, attn_map_cache):

        # diffusion loss
        diff_loss = self.get_diff_loss(model_output, input, w).mean()

        loss = diff_loss
        loss_dict = {
            "loss/diff_loss": diff_loss
        }

        # additional loss
        if self.lambda_local_loss is not None:
            local_loss = self.get_local_loss(attn_map_cache, batch).mean()
            loss += self.lambda_local_loss * local_loss
            loss_dict["loss/local_loss"] = local_loss

        if self.lambda_style_loss is not None:
            style_loss = self.get_style_loss(model_output_decoded, batch).mean()
            loss += self.lambda_style_loss * style_loss
            loss_dict["loss/style_loss"] = style_loss
        
        if self.lambda_ocr_loss is not None:
            ocr_loss = self.get_ocr_loss(model_output_decoded, batch).mean()
            loss += self.lambda_ocr_loss * ocr_loss
            loss_dict["loss/ocr_loss"] = ocr_loss

        loss_dict["loss/full_loss"] = loss

        return loss, loss_dict
    
    def get_warped_patches(self, image, tran_mxs, size):
        """
        input:
            - image: b, 3, H, W
            - tran_mxs: b, l, 2, 3
        output:
            - s_patches: b, l, 3, s, s
        """

        b, _, H, W = image.shape
        l = tran_mxs.shape[1]
        M = torch.tensor(
            [[[2/W, 0, -1],
            [0, 2/H, -1],
            [0, 0, 1]]]
        ).tile(b,1,1).to(tran_mxs) # (b, 3, 3)
        
        warped_patches = []
        for i in range(l):
            tran_mx = tran_mxs[:,i,...] # (b, 2, 3)
            tran_mx_pad = torch.tensor([[[0, 0, 1]]]).tile(b,1,1).to(tran_mx) # (b, 1, 3)
            tran_mx = torch.cat([tran_mx, tran_mx_pad], dim=1) # (b, 3, 3)
            tran_mx = torch.linalg.inv(M @ tran_mx @ torch.linalg.inv(M)) # (b, 3, 3)

            grid = F.affine_grid(tran_mx[:,:2,:], [b, 3, H, W], align_corners=True).to(image)
            warped_patch = F.grid_sample(image, grid, align_corners=True) # (b, 3, H, W)
            warped_patch = warped_patch[:,:,:size,:size] # (b, 3, s, s)
            warped_patches.append(warped_patch)
        
        warped_patches = torch.stack(warped_patches, dim=1) # (b, l, 3, s, s)

        return warped_patches
    
    def get_style_loss(self, model_output_decoded, batch):

        tran_mxs = batch["tran_mxs"] # (b, l, 2, 3)
        s_tgt = batch["s_tgt"] # (b, l, 3, s, s)
        seg_mask = batch["seg_mask"] # (b, l)
        b, l = seg_mask.shape
        size = s_tgt.shape[-1]

        warped_patches = self.get_warped_patches(model_output_decoded, tran_mxs, size) # (b, l, 3, s, s)
        
        l1_loss = (warped_patches - s_tgt).abs().reshape(b, l, -1).mean(dim=-1) # (b, l)
        l1_loss = (l1_loss * seg_mask).sum(dim=-1) / seg_mask.sum(dim=-1) # (b,)

        warped_patches = warped_patches.reshape(-1, 3, size, size) # (bl, 3, s, s)
        s_tgt = s_tgt.reshape(-1, 3, size, size) # (bl, 3, s, s)

        lpips_loss = self.lpips_loss(warped_patches, s_tgt).reshape(b, l) # (b, l)
        lpips_loss = (lpips_loss * seg_mask).sum(dim=-1) / seg_mask.sum(dim=-1) # (b,)

        style_loss = l1_loss + lpips_loss

        return style_loss
    
    def get_ocr_loss(self, model_output_decoded, batch):

        r_bbox = batch["r_bbox"]
        label = batch["label"]

        model_output_crops = []
        for i, bbox in enumerate(r_bbox):
            m_top, m_bottom, m_left, m_right = bbox
            model_output_crops.append(model_output_decoded[i, :, m_top:m_bottom, m_left:m_right])

        loss = self.predictor.calc_loss(model_output_crops, label)

        return loss

    def get_local_loss(self, attn_map_cache, batch, eps=1.0e-6):

        seg_map = batch["seg_map"]
        seg_mask = batch["seg_mask"]

        losses = []
        for item in attn_map_cache:

            name = item["name"]
            heads = item["heads"]
            size = item["size"]
            attn_map = item["attn_map"]
            if size < self.min_attn_size: continue

            bh, n, l = attn_map.shape # (bh: batch size * heads / n: pixel length(h*w) / l: token length)
            b = bh//heads

            attn_map = attn_map.reshape((b, heads, n, l)) # (b, h, n, l)
            attn_map = attn_map.permute(0, 1, 3, 2).contiguous() # (b, h, n, l) -> (b, h, l, n)
            attn_map = attn_map.mean(dim=1) # (b, l, n)

            attn_map = attn_map.reshape((b, l, size, size)) # (b, l, s, s)
            attn_map = self.gaussian_blur(attn_map) # gaussian blur on each channel
            attn_map = attn_map.reshape((b, l, n)) # (b, l, n)

            p_seg_map = F.interpolate(seg_map, size=size) # (b, l, s, s)
            p_seg_map = p_seg_map.reshape((b, l, n)) # (b, l, n)
            n_seg_map = 1 - p_seg_map

            p_loss = (p_seg_map * attn_map).sum(dim=-1) / (p_seg_map.sum(dim=-1) + eps) # (b, l)
            n_loss = (n_seg_map * attn_map).sum(dim=-1) / (n_seg_map.sum(dim=-1) + eps) # (b, l)
            p_loss = (p_loss * seg_mask).sum(dim=-1) / seg_mask.sum(dim=-1) # (b,)
            n_loss = (n_loss * seg_mask).sum(dim=-1) / seg_mask.sum(dim=-1) # (b,)

            f_loss = n_loss - p_loss # (b,)
            losses.append(f_loss)

        loss = sum(losses)/len(losses)

        return loss
    