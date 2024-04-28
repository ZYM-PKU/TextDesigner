import pytorch_lightning as pl
import torch
from omegaconf import ListConfig, OmegaConf
from safetensors.torch import load_file as load_safetensors
from contextlib import contextmanager, nullcontext
from typing import Any, Dict, List, Tuple, Union

from ..modules import UNCONDITIONAL_CONFIG
from ..modules.diffusionmodules.wrappers import OPENAIUNETWRAPPER
from ..modules.ema import LitEma
from ..util import (
    default,
    disabled_train,
    get_obj_from_str,
    instantiate_from_config,
    append_dims
)


class DiffusionEngine(pl.LightningModule):
    def __init__(
        self,
        network_config,
        denoiser_config,
        first_stage_config,
        conditioner_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        optimizer_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        scheduler_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        loss_fn_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        network_wrapper: Union[None, str] = None,
        ckpt_path: Union[None, str] = None,
        use_ema: bool = False,
        ema_decay_rate: float = 0.9999,
        scale_factor: float = 1.0,
        disable_first_stage_autocast=False,
        num_condition: int = 1,
        input_key: str = "jpg",
        log_keys: Union[List, None] = None,
        compile_model: bool = False,
        opt_keys: Union[List, None] = None
    ):
        super().__init__()
        self.opt_keys = opt_keys
        self.log_keys = log_keys
        self.input_key = input_key
        self.num_condition = num_condition
        self.optimizer_config = default(
            optimizer_config, {"target": "torch.optim.AdamW"}
        )
        model = instantiate_from_config(network_config)
        self.model = get_obj_from_str(default(network_wrapper, OPENAIUNETWRAPPER))(
            model, compile_model=compile_model
        )

        self._init_first_stage(first_stage_config)
        self.conditioner = instantiate_from_config(
            default(conditioner_config, UNCONDITIONAL_CONFIG)
        )
        self.conditioner.init_model_from_obj(self.first_stage_model)

        self.denoiser = instantiate_from_config(denoiser_config)
        self.model.diffusion_model.init_denoiser_from_obj(self.denoiser)

        self.scheduler_config = scheduler_config

        self.loss_fn = (
            instantiate_from_config(loss_fn_config)
            if loss_fn_config is not None
            else None
        )

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model, decay=ema_decay_rate)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.scale_factor = scale_factor
        self.disable_first_stage_autocast = disable_first_stage_autocast

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)

    def init_from_ckpt(
        self,
        path: str,
    ) -> None:
        if path.endswith("ckpt"):
            sd = torch.load(path, map_location="cpu")["state_dict"]
        elif path.endswith("safetensors"):
            sd = load_safetensors(path)
        else:
            raise NotImplementedError

        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(
            f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys"
        )
        missing = [m for m in missing if m.startswith('model.diffusion_model')]
        unexpected = [u for u in unexpected if u.startswith('model.diffusion_model')]
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def freeze(self):
        for param in self.parameters():
            param.requires_grad_(False)
            
    def _init_first_stage(self, config):
        model = instantiate_from_config(config).eval()
        model.train = disabled_train
        for param in model.parameters():
            param.requires_grad = False
        self.first_stage_model = model

    def get_input(self, batch):
        # assuming unified data format, dataloader returns a dict.
        # image tensors should be scaled to -1 ... 1 and in bchw format
        return batch[self.input_key]

    def decode_first_stage(self, z, use_grad=False):
        context = nullcontext if use_grad else torch.no_grad
        with context():
            z = 1.0 / self.scale_factor * z
            with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
                out = self.first_stage_model.decode(z)
        return out

    def encode_first_stage(self, x, use_grad=False):
        context = nullcontext if use_grad else torch.no_grad
        with context():
            with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
                z = self.first_stage_model.encode(x)
            z = self.scale_factor * z
        return z

    def forward(self, x, batch):
        cond = self.conditioner(batch)

        idx = torch.randint(0, self.denoiser.num_idx, (x.shape[0],))
        sigmas = self.denoiser.idx_to_sigma(idx).to(x.device)

        noise = torch.randn_like(x)
        if self.loss_fn.offset_noise_level > 0.0:
            noise = noise + self.loss_fn.offset_noise_level * append_dims(
                torch.randn(x.shape[0], device=x.device), x.ndim
            )

        noised_input = x + noise * append_dims(sigmas, x.ndim)
        model_output = self.denoiser(self.model, noised_input, sigmas, cond)
        w = append_dims(self.denoiser.w(sigmas), x.ndim)

        model_output_decoded = None
        if self.loss_fn.lambda_style_loss is not None:
            model_output_decoded = self.decode_first_stage(model_output, use_grad=True)

        loss, loss_dict = self.loss_fn(x, model_output, w, batch, model_output_decoded,\
                                        self.model.diffusion_model.attn_map_cache)

        return loss, loss_dict

    def shared_step(self, batch: Dict) -> Any:
        x = self.get_input(batch)
        x = self.encode_first_stage(x)

        batch["global_step"] = self.global_step
        loss, loss_dict = self(x, batch)

        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)

        self.log_dict(
            loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=False
        )

        self.log(
            "global_step",
            float(self.global_step),
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False
        )

        lr = self.optimizers().param_groups[0]["lr"]
        self.log(
            "lr_abs", lr, prog_bar=True, logger=True, on_step=False, on_epoch=True
        )

        return loss
    
    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

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

    def instantiate_optimizer_from_config(self, params, lr, cfg):
        return get_obj_from_str(cfg["target"])(
            params, lr=lr, **cfg.get("params", dict())
        )

    def configure_optimizers(self):
        lr = self.learning_rate
        params = []
        print("Trainable parameter list: ")
        print("="*50)
        for name, param in self.model.named_parameters():
            if any([key in name for key in self.opt_keys]):
                params.append(param)
                print(name)
            else:
                param.requires_grad_(False)
        print("-"*50)
        for name, param in self.conditioner.named_parameters():
            if param.requires_grad:
                params.append(param)
                print(f"conditioner.{name}")
        print("="*50)
        
        opt = self.instantiate_optimizer_from_config(params, lr, self.optimizer_config)
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda epoch: 0.95**epoch)

        return [opt], scheduler
