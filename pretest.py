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
        real = batch["gt"].to(torch.device("cuda", index=cfgs.gpu))

        fake = model(c_ref, s_ref)
        fake = torch.clamp((fake + 1.0) / 2.0, min=0.0, max=1.0)
        fake = fake.cpu().numpy().transpose(0,2,3,1) * 255

        c_ref = ((c_ref+1.0)/2.0).tile(1,3,1,1).cpu().numpy().transpose(0,2,3,1) * 255
        real = ((real+1.0)/2.0).cpu().numpy().transpose(0,2,3,1) * 255

        c_ref = np.concatenate(list(c_ref), axis=0)
        real = np.concatenate(list(real), axis=0)
        fake = np.concatenate(list(fake), axis=0)
        out = np.concatenate([c_ref, real, fake], axis=1)
        out = Image.fromarray(out.astype(np.uint8))
        out.save(ospj(output_dir, f"{idx}.png"))


if __name__=='__main__':

    sys.path.append(os.getcwd())

    config_path = 'configs/pretest.yaml'
    cfgs = OmegaConf.load(config_path)

    test(cfgs)