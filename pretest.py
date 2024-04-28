import os, sys
import torch

from omegaconf import OmegaConf
from dataset import get_dataloader
from torchvision.utils import save_image
from os.path import join as ospj
from PIL import Image
from util import *


def test(cfgs):

    model = get_model(cfgs)
    dataloader = get_dataloader(cfgs)

    output_dir = cfgs.output_dir
    os.makedirs(output_dir, exist_ok=True)
    os.system(f"rm -rf {ospj(output_dir, '*')}")

    for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):

        if idx >= cfgs.max_iter: break

        c_ref = batch["c_ref"].to(torch.device("cuda", index=cfgs.gpu))
        s_ref = batch["s_ref"].to(torch.device("cuda", index=cfgs.gpu))
        c_gt = batch["c_gt"].to(torch.device("cuda", index=cfgs.gpu))
        s_gt = batch["s_gt"].to(torch.device("cuda", index=cfgs.gpu))

        c_fake, s_fake = model(c_ref, s_ref)
        c_fake = c_fake.tile(1, 3, 1, 1)
        c_fake = torch.clamp((c_fake + 1.0) / 2.0, min=0.0, max=1.0)
        s_fake = torch.clamp((s_fake + 1.0) / 2.0, min=0.0, max=1.0)
        c_fake = c_fake.cpu().numpy().transpose(0,2,3,1) * 255
        s_fake = s_fake.cpu().numpy().transpose(0,2,3,1) * 255
        
        c_ref = ((c_ref+1.0)/2.0).tile(1,3,1,1).cpu().numpy().transpose(0,2,3,1) * 255
        c_gt = ((c_gt+1.0)/2.0).tile(1,3,1,1).cpu().numpy().transpose(0,2,3,1) * 255
        s_gt = ((s_gt+1.0)/2.0).cpu().numpy().transpose(0,2,3,1) * 255

        c_ref = np.concatenate(list(c_ref), axis=0)
        c_gt = np.concatenate(list(c_gt), axis=0)
        s_gt = np.concatenate(list(s_gt), axis=0)
        c_fake = np.concatenate(list(c_fake), axis=0)
        s_fake = np.concatenate(list(s_fake), axis=0)
        out = np.concatenate([c_ref, c_gt, c_fake, s_gt, s_fake], axis=1)
        out = Image.fromarray(out.astype(np.uint8))
        out.save(ospj(output_dir, f"{idx}.png"))


if __name__=='__main__':

    sys.path.append(os.getcwd())

    config_path = 'configs/pretest.yaml'
    cfgs = OmegaConf.load(config_path)

    test(cfgs)