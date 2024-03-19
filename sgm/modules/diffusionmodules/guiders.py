from functools import partial

import torch

from ...util import default, instantiate_from_config


class VanillaCFG:
    """
    implements parallelized CFG
    """

    def __init__(self, scale, dyn_thresh_config=None):

        self.num_cond = 2
        scale_schedule = lambda scale, sigma: scale  # independent of step
        self.scale_schedule = partial(scale_schedule, scale)
        self.dyn_thresh = instantiate_from_config(
            default(
                dyn_thresh_config,
                {
                    "target": "sgm.modules.diffusionmodules.sampling_utils.NoDynamicThresholding"
                },
            )
        )

    def __call__(self, x, sigma):

        xs = x.chunk(self.num_conds)
        scale_value = self.scale_schedule(sigma)
        x_pred = self.dyn_thresh(xs, scale_value)
        
        return x_pred

    def prepare_inputs(self, x, s, conds):
        
        c = conds[0]
        c_out = dict()
        self.num_conds = len(conds)
        for k in c:
            assert isinstance(c[k], torch.Tensor)
            c_out[k] = c[k]
            for cond in conds[:0:-1]:
                c_out[k] = torch.cat((cond[k], c_out[k]), 0)

        return torch.cat([x] * self.num_conds), torch.cat([s] * self.num_conds), c_out
    

class IdentityGuider:
    def __call__(self, x, sigma):
        return x

    def prepare_inputs(self, x, s, c, uc):
        c_out = dict()

        for k in c:
            c_out[k] = c[k]

        return x, s, c_out
