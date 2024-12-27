import os
import os.path as osp
from torchvision.utils import make_grid, save_image
from torch_radon import RadonFanbeam
import argparse
import tqdm
import copy
import math
from utils.measure import *
from utils.ema import EMA

from models.basic_template import TrainTask
from .SPDiff_wrapper import Network
from .diffusion_modules import Diffusion
import time
import wandb


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class SPDiff(TrainTask):
    @staticmethod
    def build_options():
        parser = argparse.ArgumentParser('Private arguments for training of different methods')
        parser.add_argument("--in_channels", default=1, type=int)
        parser.add_argument("--out_channels", default=1, type=int)
        parser.add_argument("--init_lr", default=2e-4, type=float)

        parser.add_argument('--update_ema_iter', default=10, type=int)
        parser.add_argument('--start_ema_iter', default=2000, type=int)
        parser.add_argument('--ema_decay', default=0.995, type=float)

        parser.add_argument('--T', default=10, type=int)
        parser.add_argument('--lossfn', default='mse', type=str)
        parser.add_argument('--schedule_type', default='linear', type=str)
        return parser

    def set_model(self):
        opt = self.opt
        self.ema = EMA(opt.ema_decay)
        self.update_ema_iter = opt.update_ema_iter
        self.start_ema_iter = opt.start_ema_iter
        self.dose = opt.dose
        self.T = opt.T
        self.context = opt.context

        # schedule
        self.I0 = 2.5e5
        self.start_I0 = 2.5e4
        self.Dose = [1, 2, 4, 8, 10, 20]
        self.lambda_ = self.frac_schedule(opt.T, lambda_0=3e5, lambda_T=self.start_I0)
        self.lambda_ = self.lambda_.cuda()

        # Initialize model
        denoise_fn = Network(in_channels=opt.in_channels, context=opt.context)
        model = Diffusion(
            denoise_fn=denoise_fn,
            img_size=opt.img_size,
            timesteps=opt.T,
            context=opt.context,
            lambda_=self.lambda_
        ).cuda()

        optimizer = torch.optim.Adam(model.parameters(), opt.init_lr)
        ema_model = copy.deepcopy(model)

        self.logger.modules = [model, ema_model, optimizer]
        self.model = model
        self.optimizer = optimizer
        self.ema_model = ema_model
        self.lossfn = nn.L1Loss(reduction='mean')
        self.reset_parameters()

        # Projection geometry and reconstruction parameters
        image_size = 512
        detect_number = 672
        angle = 672
        src_dist = 1361.2 - 615.18
        det_dist = 615.18
        spacing = 1.85 / 0.68
        angles = np.linspace(-1 / 2 * np.pi, 3 / 2 * np.pi, angle).astype(np.float32)
        self.u_water = 0.0192
        self.kappa = 1.5
        self.radon = RadonFanbeam(
            image_size,
            angles,
            src_dist,
            det_dist,
            det_count=detect_number,
            det_spacing=spacing,
            clip_to_circle=False
        )

    def frac_schedule(self, timesteps, lambda_0=3e5, lambda_T=2.5e4):
        steps = timesteps
        scale = lambda_0 / lambda_T
        x = 1 / torch.linspace(1, scale, steps)
        lambda_ = x * lambda_0
        return lambda_

    def q_sample(self, x_start, t):
        lambda_t = extract(self.lambda_, t, x_start.shape)
        print(lambda_t)
        return torch.poisson(lambda_t * x_start + 10) / lambda_t

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self, n_iter):
        if n_iter < self.start_ema_iter:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def calculate_match_t(self, ld_I0):
        min_diff = 3e5
        for i in range(self.lambda_.shape[0]):
            diff = torch.abs(self.lambda_[i] - ld_I0)
            if diff < min_diff:
                min_diff = diff
                match_t = i
        return match_t

    def train(self, inputs, n_iter):
        opt = self.opt
        self.model.train()
        self.ema_model.train()
        low_dose, full_dose = inputs
        low_dose, full_dose  = low_dose.cuda(), full_dose.cuda()

        gen_full_dose, x_mix = self.model(full_dose)

        if self.context:
            full_dose = full_dose[:, 1].unsqueeze(1)
        loss = self.lossfn(gen_full_dose, full_dose)
        loss.backward()

        # Initialize wandb
        if opt.wandb:
            if n_iter == opt.resume_iter + 1:
                wandb.init(project="NEED")

        self.optimizer.step()
        self.optimizer.zero_grad()

        lr = self.optimizer.param_groups[0]['lr']

        loss = loss.item()
        self.logger.msg([loss, lr], n_iter)
        if opt.wandb:
            wandb.log({'epoch': n_iter, 'loss': loss})

        if n_iter % self.update_ema_iter == 0:
            self.step_ema(n_iter)

    @torch.no_grad()
    def test(self, n_iter, up_log=True, save_data=False):
        opt = self.opt
        self.ema_model.eval()

        psnr, ssim, rmse = 0., 0., 0.
        if opt.test_dataset == 'mayo2020':
            # abdomen
            ld_I0_a = self.I0 / 4
            match_t_a = self.calculate_match_t(ld_I0_a)
            # chest
            ld_I0_c = self.I0 / 10
            match_t_c = self.calculate_match_t(ld_I0_c)
        else:
            ld_I0 = self.I0 / self.Dose[opt.dose]
            match_t = self.calculate_match_t(ld_I0)

        idx = 0
        gen_full_doses = []
        for low_dose, full_dose in tqdm.tqdm(self.test_loader, desc='test'):
            if self.context:
                full_dose = full_dose[:, 1].unsqueeze(1)

            if opt.test_dataset == 'mayo2020':
                if idx < 400:
                    match_t = match_t_c
                else:
                    match_t = match_t_a

            low_dose, full_dose = low_dose.cuda(), full_dose.cuda()
            gen_full_dose, _, _ = self.ema_model.sample(
                batch_size = 1,
                img = low_dose,
                t = match_t + 1
            )
            idx += 1

            if save_data:
                gen_full_dose_save = self.transfer_HU(gen_full_dose)
                gen_full_doses.append(gen_full_dose_save)
            gen_full_dose = self.recon(gen_full_dose, self.radon, self.u_water, self.kappa)
            full_dose = self.recon(full_dose, self.radon, self.u_water, self.kappa)
            full_dose = self.transfer_calculate_window(full_dose)
            gen_full_dose = self.transfer_calculate_window(gen_full_dose)

            data_range = full_dose.max() - full_dose.min()
            psnr_score, ssim_score, rmse_score = compute_measure(full_dose, gen_full_dose, data_range)
            psnr += psnr_score / len(self.test_loader)
            ssim += ssim_score / len(self.test_loader)
            rmse += rmse_score / len(self.test_loader)

        self.logger.msg([psnr, ssim, rmse], n_iter)
        if up_log and opt.wandb:
            wandb.log({'epoch': n_iter, 'PSNR': psnr, 'SSIM': ssim, 'RMSE': rmse})

        if save_data:
            if opt.test_dataset == 'mayo2020':
                save_root = osp.join(opt.npy_root, 'NEED/Dataset/phase_one/data_mayo2020')
                os.makedirs(save_root, exist_ok=True)
                file_name = osp.join(save_root, opt.model_name + '_{}.npy'.format(opt.test_iter))
                np.save(file_name, torch.stack(gen_full_doses).squeeze_().cpu().detach().numpy())
            else:
                save_root = osp.join(opt.npy_root, 'NEED/Dataset/phase_one/data_mayo2016')
                os.makedirs(save_root, exist_ok=True)
                file_name = osp.join(save_root, opt.model_name + '_{}_{}.npy'.format(opt.test_iter, opt.dose))
                np.save(file_name, torch.stack(gen_full_doses).squeeze_().cpu().detach().numpy())

    @torch.no_grad()
    def generate_images(self, n_iter):
        opt = self.opt
        self.ema_model.eval()
        low_dose, full_dose = self.test_images

        if opt.test_dataset == 'mayo2020':
            # abdomen
            ld_I0_a = self.I0 / 4
            match_t_a = self.calculate_match_t(ld_I0_a)
            # chest
            ld_I0_c = self.I0 / 10
            match_t_c = self.calculate_match_t(ld_I0_c)
        else:
            ld_I0 = self.I0 / self.Dose[opt.dose]
            match_t = self.calculate_match_t(ld_I0)

        gen_full_doses, direct_recons, imstep_imgs = [], [], []
        for i in range(low_dose.shape[0]):
            if opt.test_dataset == 'mayo2020':
                if i < 2:
                    match_t = match_t_c
                else:
                    match_t = match_t_a

            gen_full_dose, direct_recon, imstep_img = self.ema_model.sample(
                batch_size=1,
                img=low_dose[i].unsqueeze(0),
                t=match_t + 1
            )
            gen_full_doses.append(gen_full_dose)

        gen_full_dose = torch.stack(gen_full_doses).squeeze(1)

        if self.context:
            low_dose = low_dose[:, 1].unsqueeze(1)
            full_dose = full_dose[:, 1].unsqueeze(1)

        low_dose = self.recon(low_dose, self.radon, self.u_water, self.kappa)
        full_dose = self.recon(full_dose, self.radon, self.u_water, self.kappa)
        gen_full_dose = self.recon(gen_full_dose, self.radon, self.u_water, self.kappa)

        target = self.transfer_calculate_window(full_dose)
        output = self.transfer_calculate_window(gen_full_dose)

        data_range = target.max() - target.min()
        psnr_score, ssim_score, rmse_score = compute_measure(target, output, data_range)
        print('{:.2f}, {:.4f}, {:.2f}'.format(psnr_score, ssim_score, rmse_score))

        b, c, w, h = low_dose.size()
        fake_imgs = torch.stack([low_dose, full_dose, gen_full_dose])
        fake_imgs = self.transfer_display_window(fake_imgs)
        fake_imgs = fake_imgs.transpose(1, 0).reshape((-1, c, w, h))
        self.logger.save_image(make_grid(fake_imgs, nrow=3), n_iter, 'test_{}'.format(self.dose))
