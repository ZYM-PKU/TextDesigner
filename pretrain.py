from omegaconf import OmegaConf

import os,sys
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from dataset import get_dataloader
from util import *

import logging
logging.basicConfig(level=logging.CRITICAL) # silence the TTFont warning


def train(cfgs):

    dataloader = get_dataloader(cfgs)
    model = get_model(cfgs)

    checkpoint_callback = ModelCheckpoint(dirpath = cfgs.ckpt_dir, every_n_epochs = cfgs.check_freq)

    trainer = pl.Trainer(callbacks = [checkpoint_callback], **cfgs.lightning)
    trainer.fit(model = model, train_dataloaders = dataloader)

    
if __name__ == "__main__":

    sys.path.append(os.getcwd())
    
    # global settings
    torch.set_float32_matmul_precision('medium') # matrix multiply precision

    config_path = 'configs/pretrain.yaml'
    cfgs = OmegaConf.load(config_path)
    train(cfgs)