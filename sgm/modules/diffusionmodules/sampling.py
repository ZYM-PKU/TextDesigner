"""
    Partially ported from https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/sampling.py
"""


from typing import Dict, Union

import imageio
import torch
import numpy as np
import torch.nn.functional as F
from omegaconf import ListConfig, OmegaConf
from tqdm import tqdm

from ...modules.diffusionmodules.sampling_utils import (
    get_ancestral_step,
    linear_multistep_coeff,
    to_d,
    to_neg_log_sigma,
    to_sigma,
)
from ...util import append_dims, default, instantiate_from_config
from torchvision.utils import save_image

DEFAULT_GUIDER = {"target": "sgm.modules.diffusionmodules.guiders.IdentityGuider"}


class BaseDiffusionSampler:
    def __init__(
        self,
        discretization_config: Union[Dict, ListConfig, OmegaConf],
        num_steps: Union[int, None] = None,
        guider_config: Union[Dict, ListConfig, OmegaConf, None] = None,
        verbose: bool = False,
        device: str = "cuda",
    ):
        self.num_steps = num_steps
        self.discretization = instantiate_from_config(discretization_config)
        self.guider = instantiate_from_config(
            default(
                guider_config,
                DEFAULT_GUIDER,
            )
        )
        self.verbose = verbose
        self.device = device

    def prepare_sampling_loop(self, x, cond, uc=None, num_steps=None):
        sigmas = self.discretization(
            self.num_steps if num_steps is None else num_steps, device=self.device
        )
        uc = default(uc, cond)

        x *= torch.sqrt(1.0 + sigmas[0] ** 2.0)
        num_sigmas = len(sigmas)

        s_in = x.new_ones([x.shape[0]])

        return x, s_in, sigmas, num_sigmas, cond, uc

    def denoise(self, x, model, sigma, cond, uc):
        denoised = model.denoiser(model.model, *self.guider.prepare_inputs(x, sigma, cond, uc))
        denoised = self.guider(denoised, sigma)
        return denoised

    def get_sigma_gen(self, num_sigmas, init_step=0):
        sigma_generator = range(init_step, num_sigmas - 1)
        if self.verbose:
            print("#" * 30, " Sampling setting ", "#" * 30)
            print(f"Sampler: {self.__class__.__name__}")
            print(f"Discretization: {self.discretization.__class__.__name__}")
            print(f"Guider: {self.guider.__class__.__name__}")
            sigma_generator = tqdm(
                sigma_generator,
                total=num_sigmas-1-init_step,
                desc=f"Sampling with {self.__class__.__name__} for {num_sigmas-1-init_step} steps",
            )
        return sigma_generator


class SingleStepDiffusionSampler(BaseDiffusionSampler):
    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc, *args, **kwargs):
        raise NotImplementedError

    def euler_step(self, x, d, dt):
        return x + dt * d
    

class DDIMSampler(BaseDiffusionSampler):

    def prepare_sampling_loop(self, x, num_steps=None):

        timesteps, alphas_cumprod = self.discretization.get_schedule(
            self.num_steps if num_steps is None else num_steps
        )

        timesteps = np.flip(timesteps, 0)
        timesteps = torch.from_numpy(timesteps).to(device=self.device)
        alphas_cumprod = torch.from_numpy(alphas_cumprod).to(device=self.device)

        return x, timesteps, alphas_cumprod

    def get_init_noise(self, cfgs):

        shape = (cfgs.batch_size, cfgs.channel, int(cfgs.H) // cfgs.factor, int(cfgs.W) // cfgs.factor)

        randn = torch.randn(shape).to(torch.device("cuda", index=cfgs.gpu))

        return randn

    def sampler_step(self, timestep, alphas_cumprod, x, model, conds, batch=None, save_attn=False):

        x_input, sigma, cond = self.guider.prepare_inputs(x, torch.zeros(1), conds)

        model_output = model.model(x_input, timestep.tile(x_input.shape[0]), cond)
        model_output = self.guider(model_output, None)

        if save_attn:
            model.model.diffusion_model.save_attn_map(save_name=batch["name"][0], tokens=batch["label"][0])

        prev_timestep = timestep - self.discretization.num_timesteps // self.num_steps

        alpha_prod_t = alphas_cumprod[timestep]
        alpha_prod_t_prev = alphas_cumprod[prev_timestep] if prev_timestep >= 0 else alphas_cumprod[0]
        beta_prod_t = 1 - alpha_prod_t

        pred_original_sample = (x - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        pred_epsilon = model_output

        pred_sample_direction = (1 - alpha_prod_t_prev) ** (0.5) * pred_epsilon
        prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
        
        return prev_sample

    def __call__(self, model, x, conds, batch=None, num_steps=None, detailed=False, *args, **kwargs):

        x, timesteps, alphas_cumprod = self.prepare_sampling_loop(x, num_steps)

        for i in self.get_sigma_gen(self.num_steps+1):

            timestep = timesteps[i]

            x = self.sampler_step(
                timestep,
                alphas_cumprod,
                x,
                model,
                conds,
                batch,
                save_attn=detailed and (i == self.num_steps//2)
            )

        return x
    

class DDIMInvSampler(BaseDiffusionSampler):

    def prepare_sampling_loop(self, x, cond, uc=None, num_steps=None):
        timesteps, alphas_cumprod = self.discretization.get_schedule(
            self.num_steps if num_steps is None else num_steps
        )
        uc = default(uc, cond)

        timesteps = np.ascontiguousarray(timesteps)
        timesteps = torch.from_numpy(timesteps).to(device=self.device)
        alphas_cumprod = torch.from_numpy(alphas_cumprod).to(device=self.device)

        return x, timesteps, alphas_cumprod, cond, uc

    def sampler_step(self, timestep, alphas_cumprod, x, model, cond, uc):

        x_input, sigma, cond = self.guider.prepare_inputs(x, torch.zeros(1), cond, uc)

        model_output = model.model(x_input, timestep.tile(x_input.shape[0]), cond)
        model_output = self.guider(model_output, None)

        prev_timestep = timestep
        timestep = timestep - self.discretization.num_timesteps // self.num_steps

        alpha_prod_t = alphas_cumprod[timestep] if timestep >= 0 else alphas_cumprod[0]
        alpha_prod_t_prev = alphas_cumprod[prev_timestep]
        beta_prod_t = 1 - alpha_prod_t

        pred_original_sample = (x - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        pred_epsilon = model_output

        pred_sample_direction = (1 - alpha_prod_t_prev) ** (0.5) * pred_epsilon
        prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
        
        return prev_sample

    def __call__(self, model, x, cond, uc=None, num_steps=None, *args, **kwargs):
        x, timesteps, alphas_cumprod, cond, uc = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )

        latents = [x]
        for i in self.get_sigma_gen(self.num_steps+1):

            timestep = timesteps[i]

            x = self.sampler_step(
                timestep,
                alphas_cumprod,
                x,
                model,
                cond,
                uc
            )

            latents.append(x)
        
        latents.reverse()

        return latents


class EDMSampler(SingleStepDiffusionSampler):
    def __init__(
        self, s_churn=0.0, s_tmin=0.0, s_tmax=float("inf"), s_noise=1.0, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.s_churn = s_churn
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax
        self.s_noise = s_noise

    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc=None, gamma=0.0):
        sigma_hat = sigma * (gamma + 1.0)
        if gamma > 0:
            eps = torch.randn_like(x) * self.s_noise
            x = x + eps * append_dims(sigma_hat**2 - sigma**2, x.ndim) ** 0.5

        denoised = self.denoise(x, denoiser, sigma_hat, cond, uc)
        d = to_d(x, sigma_hat, denoised)
        dt = append_dims(next_sigma - sigma_hat, x.ndim)

        euler_step = self.euler_step(x, d, dt)
        x = self.possible_correction_step(
            euler_step, x, d, dt, next_sigma, denoiser, cond, uc
        )
        return x

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )

        for i in self.get_sigma_gen(num_sigmas):
            gamma = (
                min(self.s_churn / (num_sigmas - 1), 2**0.5 - 1)
                if self.s_tmin <= sigmas[i] <= self.s_tmax
                else 0.0
            )
            x = self.sampler_step(
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x,
                cond,
                uc,
                gamma,
            )

        return x


class EulerEDMSampler(EDMSampler):

    def possible_correction_step(
        self, euler_step, x, d, dt, next_sigma, denoiser, cond, uc
    ):
        return euler_step

    def get_c_noise(self, x, model, sigma):
        sigma = model.denoiser.possibly_quantize_sigma(sigma)
        sigma_shape = sigma.shape
        sigma = append_dims(sigma, x.ndim)
        c_skip, c_out, c_in, c_noise = model.denoiser.scaling(sigma)
        c_noise = model.denoiser.possibly_quantize_c_noise(c_noise.reshape(sigma_shape))
        return c_noise
    
    def attend_and_excite(self, x, model, sigma, cond, batch, alpha, iter_enabled, thres, max_iter=20):

        # calc timestep
        c_noise = self.get_c_noise(x, model, sigma)
        
        x = x.clone().detach().requires_grad_(True)  # https://github.com/yuval-alaluf/Attend-and-Excite/blob/main/pipeline_attend_and_excite.py#L288

        iters = 0
        while True:

            model_output = model.model(x, c_noise, cond)
            local_loss = model.loss_fn.get_min_local_loss(model.model.diffusion_model.attn_map_cache, batch["mask"], batch["seg_mask"])
            grad = torch.autograd.grad(local_loss.requires_grad_(True), [x], retain_graph=True)[0]
            x = x - alpha * grad
            iters += 1

            if not iter_enabled or local_loss <= thres or iters > max_iter:
                break

        return x

    def save_segment_map(self, attn_maps, tokens=None, save_name=None):

        sections = []
        for i in range(len(tokens)): 
            attn_map = attn_maps[i]
            sections.append(attn_map)
        
        section = np.stack(sections)
        np.save(f"./temp/seg_map/seg_{save_name}.npy", section)

    def get_init_noise(self, cfgs, model, cond, batch, uc=None):

        H, W = batch["target_size_as_tuple"][0]
        shape = (cfgs.batch_size, cfgs.channel, int(H) // cfgs.factor, int(W) // cfgs.factor)

        randn = torch.randn(shape).to(torch.device("cuda", index=cfgs.gpu))
        x = randn.clone()

        xs = []
        self.verbose = False
        for _ in range(cfgs.noise_iters):
            
            x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
                x, cond, uc, num_steps=2
            )

            superv = {
                "mask": batch["mask"] if "mask" in batch else None,
                "seg_mask": batch["seg_mask"] if "seg_mask" in batch else None
            }

            local_losses = []

            for i in self.get_sigma_gen(num_sigmas):

                gamma = (
                    min(self.s_churn / (num_sigmas - 1), 2**0.5 - 1)
                    if self.s_tmin <= sigmas[i] <= self.s_tmax
                    else 0.0
                )

                x, inter, local_loss = self.sampler_step(
                    s_in * sigmas[i],
                    s_in * sigmas[i + 1],
                    model,
                    x,
                    cond,
                    superv,
                    uc,
                    gamma,
                    save_loss=True
                )

                local_losses.append(local_loss.item())
            
            xs.append((randn, local_losses[-1]))

            randn = torch.randn(shape).to(torch.device("cuda", index=cfgs.gpu))
            x = randn.clone()

        self.verbose = True
        
        xs.sort(key = lambda x: x[-1])

        if len(xs) > 0:
            print(f"Init local loss: Best {xs[0][1]} Worst {xs[-1][1]}")
            x = xs[0][0]

        return x

    def sampler_step(self, sigma, next_sigma, model, x, cond, batch=None, uc=None, 
                     gamma=0.0, alpha=0, iter_enabled=False, thres=None, update=False,
                     name=None, save_loss=False, save_attn=False, save_inter=False):
        
        sigma_hat = sigma * (gamma + 1.0)
        if gamma > 0:
            eps = torch.randn_like(x) * self.s_noise
            x = x + eps * append_dims(sigma_hat**2 - sigma**2, x.ndim) ** 0.5
        
        if update:
            x = self.attend_and_excite(x, model, sigma_hat, cond, batch, alpha, iter_enabled, thres)

        denoised = self.denoise(x, model, sigma_hat, cond, uc)
        denoised_decode = model.decode_first_stage(denoised) if save_inter else None
        
        if save_loss:
            local_loss = model.loss_fn.get_min_local_loss(model.model.diffusion_model.attn_map_cache, batch["mask"], batch["seg_mask"])
            local_loss = local_loss[local_loss.shape[0]//2:]
        else:
            local_loss = torch.zeros(1)
        if save_attn:
            attn_map = model.model.diffusion_model.save_attn_map(save_name=name, tokens=batch["label"][0])
            self.save_segment_map(attn_map, tokens=batch["label"][0], save_name=name)

        d = to_d(x, sigma_hat, denoised)
        dt = append_dims(next_sigma - sigma_hat, x.ndim)

        euler_step = self.euler_step(x, d, dt)

        return euler_step, denoised_decode, local_loss
    
    def __call__(self, model, x, cond, batch=None, uc=None, num_steps=None, init_step=0, 
                 name=None, aae_enabled=False, detailed=False):

        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )

        name = batch["name"][0]
        inters = []
        local_losses = []
        scales = np.linspace(start=1.0, stop=0, num=num_sigmas)
        iter_lst = np.linspace(start=5, stop=25, num=6, dtype=np.int32)
        thres_lst = np.linspace(start=-0.5, stop=-0.8, num=6)

        for i in self.get_sigma_gen(num_sigmas, init_step=init_step):

            gamma = (
                min(self.s_churn / (num_sigmas - 1), 2**0.5 - 1)
                if self.s_tmin <= sigmas[i] <= self.s_tmax
                else 0.0
            )

            alpha = 20 * np.sqrt(scales[i])
            update = aae_enabled
            save_loss = aae_enabled
            save_attn = detailed and (i == (num_sigmas-1)//2)
            save_inter = aae_enabled

            if i in iter_lst:
                iter_enabled = True
                thres = thres_lst[list(iter_lst).index(i)]
            else:
                iter_enabled = False
                thres = 0.0

            x, inter, local_loss = self.sampler_step(
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                model,
                x,
                cond,
                batch,
                uc,
                gamma,
                alpha=alpha,
                iter_enabled=iter_enabled,
                thres=thres,
                update=update,
                name=name,
                save_loss=save_loss,
                save_attn=save_attn,
                save_inter=save_inter
            )

            local_losses.append(local_loss.item())
            if inter is not None:
                inter = torch.clamp((inter + 1.0) / 2.0, min=0.0, max=1.0)[0]
                inter = inter.cpu().numpy().transpose(1, 2, 0) * 255
                inters.append(inter.astype(np.uint8))

        print(f"Local losses: {local_losses}")

        if len(inters) > 0:
            imageio.mimsave(f"./temp/inters/{name}.gif", inters, 'GIF', duration=0.02)

        return x


class EulerEDMDualSampler(EulerEDMSampler):

    def prepare_sampling_loop(self, x, cond, uc_1=None, uc_2=None, num_steps=None):
        sigmas = self.discretization(
            self.num_steps if num_steps is None else num_steps, device=self.device
        )
        uc_1 = default(uc_1, cond)
        uc_2 = default(uc_2, cond)

        x *= torch.sqrt(1.0 + sigmas[0] ** 2.0)
        num_sigmas = len(sigmas)

        s_in = x.new_ones([x.shape[0]])

        return x, s_in, sigmas, num_sigmas, cond, uc_1, uc_2

    def denoise(self, x, model, sigma, cond, uc_1, uc_2):
        denoised = model.denoiser(model.model, *self.guider.prepare_inputs(x, sigma, cond, uc_1, uc_2))
        denoised = self.guider(denoised, sigma)
        return denoised
    
    def get_init_noise(self, cfgs, model, cond, batch, uc_1=None, uc_2=None):

        H, W = batch["target_size_as_tuple"][0]
        shape = (cfgs.batch_size, cfgs.channel, int(H) // cfgs.factor, int(W) // cfgs.factor)

        randn = torch.randn(shape).to(torch.device("cuda", index=cfgs.gpu))
        x = randn.clone()

        xs = []
        self.verbose = False
        for _ in range(cfgs.noise_iters):
            
            x, s_in, sigmas, num_sigmas, cond, uc_1, uc_2 = self.prepare_sampling_loop(
                x, cond, uc_1, uc_2, num_steps=2
            )

            superv = {
                "mask": batch["mask"] if "mask" in batch else None,
                "seg_mask": batch["seg_mask"] if "seg_mask" in batch else None
            }

            local_losses = []

            for i in self.get_sigma_gen(num_sigmas):

                gamma = (
                    min(self.s_churn / (num_sigmas - 1), 2**0.5 - 1)
                    if self.s_tmin <= sigmas[i] <= self.s_tmax
                    else 0.0
                )

                x, inter, local_loss = self.sampler_step(
                    s_in * sigmas[i],
                    s_in * sigmas[i + 1],
                    model,
                    x,
                    cond,
                    superv,
                    uc_1,
                    uc_2,
                    gamma,
                    save_loss=True
                )

                local_losses.append(local_loss.item())
            
            xs.append((randn, local_losses[-1]))

            randn = torch.randn(shape).to(torch.device("cuda", index=cfgs.gpu))
            x = randn.clone()

        self.verbose = True
        
        xs.sort(key = lambda x: x[-1])

        if len(xs) > 0:
            print(f"Init local loss: Best {xs[0][1]} Worst {xs[-1][1]}")
            x = xs[0][0]

        return x

    def sampler_step(self, sigma, next_sigma, model, x, cond, batch=None, uc_1=None, uc_2=None,
                     gamma=0.0, alpha=0, iter_enabled=False, thres=None, update=False,
                     name=None, save_loss=False, save_attn=False, save_inter=False):
        
        sigma_hat = sigma * (gamma + 1.0)
        if gamma > 0:
            eps = torch.randn_like(x) * self.s_noise
            x = x + eps * append_dims(sigma_hat**2 - sigma**2, x.ndim) ** 0.5
        
        if update:
            x = self.attend_and_excite(x, model, sigma_hat, cond, batch, alpha, iter_enabled, thres)

        denoised = self.denoise(x, model, sigma_hat, cond, uc_1, uc_2)
        denoised_decode = model.decode_first_stage(denoised) if save_inter else None
        
        if save_loss:
            local_loss = model.loss_fn.get_min_local_loss(model.model.diffusion_model.attn_map_cache, batch["mask"], batch["seg_mask"])
            local_loss = local_loss[-local_loss.shape[0]//3:]
        else:
            local_loss = torch.zeros(1)
        if save_attn:
            attn_map = model.model.diffusion_model.save_attn_map(save_name=name, tokens=batch["label"][0])
            self.save_segment_map(attn_map, tokens=batch["label"][0], save_name=name)

        d = to_d(x, sigma_hat, denoised)
        dt = append_dims(next_sigma - sigma_hat, x.ndim)

        euler_step = self.euler_step(x, d, dt)

        return euler_step, denoised_decode, local_loss
    
    def __call__(self, model, x, cond, batch=None, uc_1=None, uc_2=None, num_steps=None, init_step=0, 
                 name=None, aae_enabled=False, detailed=False):

        x, s_in, sigmas, num_sigmas, cond, uc_1, uc_2 = self.prepare_sampling_loop(
            x, cond, uc_1, uc_2, num_steps
        )

        name = batch["name"][0]
        inters = []
        local_losses = []
        scales = np.linspace(start=1.0, stop=0, num=num_sigmas)
        iter_lst = np.linspace(start=5, stop=25, num=6, dtype=np.int32)
        thres_lst = np.linspace(start=-0.5, stop=-0.8, num=6)

        for i in self.get_sigma_gen(num_sigmas, init_step=init_step):

            gamma = (
                min(self.s_churn / (num_sigmas - 1), 2**0.5 - 1)
                if self.s_tmin <= sigmas[i] <= self.s_tmax
                else 0.0
            )

            alpha = 20 * np.sqrt(scales[i])
            update = aae_enabled
            save_loss = aae_enabled
            save_attn = detailed and (i == (num_sigmas-1)//2)
            save_inter = aae_enabled

            if i in iter_lst:
                iter_enabled = True
                thres = thres_lst[list(iter_lst).index(i)]
            else:
                iter_enabled = False
                thres = 0.0

            x, inter, local_loss = self.sampler_step(
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                model,
                x,
                cond,
                batch,
                uc_1,
                uc_2,
                gamma,
                alpha=alpha,
                iter_enabled=iter_enabled,
                thres=thres,
                update=update,
                name=name,
                save_loss=save_loss,
                save_attn=save_attn,
                save_inter=save_inter
            )

            local_losses.append(local_loss.item())
            if inter is not None:
                inter = torch.clamp((inter + 1.0) / 2.0, min=0.0, max=1.0)[0]
                inter = inter.cpu().numpy().transpose(1, 2, 0) * 255
                inters.append(inter.astype(np.uint8))
        
        print(f"Local losses: {local_losses}")

        if len(inters) > 0:
            imageio.mimsave(f"./temp/inters/{name}.gif", inters, 'GIF', duration=0.02)

        return x