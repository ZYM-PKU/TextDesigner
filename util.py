import torch
import copy
from omegaconf import OmegaConf
from sgm.util import instantiate_from_config
from sgm.modules.diffusionmodules.sampling import *


def init_model(cfgs):

    model_cfg = OmegaConf.load(cfgs.model_cfg_path)
    if cfgs.datype != "train":
        model_cfg.model.params.loss_fn_config = None
    model = instantiate_from_config(model_cfg.model)

    ckpt = cfgs.load_ckpt_path
    model.init_from_ckpt(ckpt)

    if cfgs.datype == "train":
        model.train()
    else:
        model.to(torch.device("cuda", index=cfgs.gpu))
        model.eval()
        model.freeze()

    return model

def get_model(cfgs):

    model = instantiate_from_config(cfgs.model)
    if cfgs.datype == "train":
        model.train()
    else:
        model.to(torch.device("cuda", index=cfgs.gpu))
        model.eval()
        model.freeze()

    return model

def init_sampling(cfgs):

    sampler = instantiate_from_config(cfgs.sampler_cfg)
    sampler.device = torch.device("cuda", index=cfgs.gpu)

    return sampler

def deep_copy(batch):

    c_batch = {}
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            c_batch[key] = torch.clone(batch[key])
        elif isinstance(batch[key], (list, dict, set)): 
            c_batch[key] = copy.deepcopy(batch[key])
        else:
            c_batch[key] = batch[key]
    
    return c_batch