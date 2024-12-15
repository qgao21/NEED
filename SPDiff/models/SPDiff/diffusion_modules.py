import torch
from torch import nn
from torchvision.utils import save_image, make_grid
from torch_radon import RadonFanbeam
import math
import copy
import numpy as np


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class Diffusion(nn.Module):
    def __init__(self,
        denoise_fn = None,
        img_size = 512,
        channels = 1,
        timesteps = 10,
        context = True,
        lambda_ = None
    ):
        super().__init__()
        self.channels = channels
        self.img_size = img_size
        self.denoise_fn = denoise_fn   # network
        self.num_timesteps = int(timesteps)
        self.context = context
        self.register_buffer('lambda_', lambda_)

    def q_sample(self, x_start, t):
        lambda_t = extract(self.lambda_, t, x_start.shape)
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x_start.shape) - 1))))  # no noise when t == 0
        return nonzero_mask * torch.poisson(lambda_t * x_start + 10) / lambda_t + (1-nonzero_mask) * x_start

    @torch.no_grad()
    def sample(self, batch_size=16, img=None, t=None):
        self.denoise_fn.eval()
        if t == None:
            t = self.num_timesteps

        if self.context:
            up_img = img[:, 0].unsqueeze(1)
            down_img = img[:, 2].unsqueeze(1)
            img = img[:, 1].unsqueeze(1)

        direct_recons = []
        imstep_imgs = []

        while (t):
            step = torch.full((batch_size,), t - 1, dtype=torch.long, device=img.device)
            if self.context:
                full_img = torch.cat((up_img, img, down_img), dim=1)
            else:
                full_img = img
            pred_x0 = self.denoise_fn(full_img, step)
            direct_recons.append(pred_x0)

            if t > 2:
                lambda_t = extract(self.lambda_, step, img.shape)
                lambda_t_sub1 = extract(self.lambda_, step - 1, img.shape)
                tau_t = torch.poisson((lambda_t_sub1 - lambda_t) * pred_x0)
                img = (1 / lambda_t_sub1) * (lambda_t * img + tau_t)
            else:
                img = pred_x0
            imstep_imgs.append(img)
            t = t - 1
        return img.clamp(0., 1.), torch.stack(direct_recons), torch.stack(imstep_imgs)

    def forward(self, y):
        b, c, h, w, device, img_size, = *y.shape, y.device, self.img_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'

        # t_single = torch.randint(0, self.num_timesteps, (1,), device=device).long()
        # t = t_single.repeat((b,))
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        if self.context:
            x_mix = self.q_sample(x_start=y[:, 1].unsqueeze(1), t=t)
            x_mix_up = self.q_sample(x_start=y[:, 0].unsqueeze(1), t=t)
            x_mix_down = self.q_sample(x_start=y[:, 2].unsqueeze(1), t=t)
            x_mix = torch.cat((x_mix_up, x_mix, x_mix_down), dim=1)
        else:
            x_mix = self.q_sample(x_start=y, t=t)

        x_recon = self.denoise_fn(x_mix, t)

        return x_recon, x_mix