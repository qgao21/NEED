import os
import torch.utils.data
import torch
import os.path as osp
import tqdm
import copy
import argparse
import torch.distributed as dist

from utils.dataset import dataset_dict
from utils.loggerx import LoggerX
from utils.sampler import RandomSampler
from utils.ops import load_network
import wandb


class TrainTask(object):

    def __init__(self, opt):
        self.opt = opt
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
        self.save_root = osp.join(
            osp.dirname(osp.dirname(osp.abspath(__file__))), 'output', '{}_{}'.format(opt.model_name, opt.run_name))
        self.logger = LoggerX(save_root=self.save_root)
        self.rank = dist.get_rank() if dist.is_initialized() else 0

        self.set_loader()
        self.set_model()

    @staticmethod
    def build_default_options():
        parser = argparse.ArgumentParser('Default arguments for training of different methods')
        parser.add_argument('--gpu_id', type=str, default='0')

        parser.add_argument('--save_freq', type=int, default=2500,
                            help='save frequency')
        parser.add_argument('--batch_size', type=int, default=8,
                            help='batch_size')
        parser.add_argument('--test_batch_size', type=int, default=1,
                            help='test_batch_size')
        parser.add_argument('--num_workers', type=int, default=4,
                            help='num of workers to use')
        parser.add_argument('--max_iter', type=int, default=100000,
                            help='number of training epochs')
        parser.add_argument('--resume_iter', type=int, default=0,
                            help='number of training epochs')
        parser.add_argument('--test_iter', type=int, default=100000,
                            help='number of epochs for test')
        parser.add_argument("--local_rank", type=int, default=0)
        parser.add_argument("--mode", type=str, default='train')
        parser.add_argument('--wandb', action="store_true")

        # optimization
        parser.add_argument('--network', type=str, default='cbdnet')
        parser.add_argument('--use_ema', action='store_true')

        # run_name and model_name
        parser.add_argument('--run_name', type=str, default='default',
                            help='each run name')
        parser.add_argument('--model_name', type=str, default='default',
                            help='the name of method')

        # dataset
        parser.add_argument('--train_dataset', type=str, default='train')
        parser.add_argument('--test_dataset', type=str, default='test')
        parser.add_argument('--dose', type=int, default=2)
        parser.add_argument('--context', action="store_true")
        parser.add_argument('--data_type', type=str, default='img')
        parser.add_argument('--norm_min', type=float, default=-1024)
        parser.add_argument('--norm_max', type=float, default=3072)
        parser.add_argument('--img_size', type=int, default=512)

        return parser

    @staticmethod
    def build_options():
        pass

    def load_pretrained_dict(self, file_name: str):
        self.project_root = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
        return load_network(osp.join(self.project_root, 'pretrained', file_name))

    def set_loader(self):
        opt = self.opt
        opt.npy_root = 'your root path'

        train_dataset = dataset_dict[opt.train_dataset](
            npy_root = opt.npy_root,
            dose = opt.dose,
            context = opt.context,
            data_type = opt.data_type,
            norm_min = opt.norm_min,
            norm_max = opt.norm_max,
        )
        test_dataset = dataset_dict[opt.test_dataset](
            npy_root=opt.npy_root,
            dose = opt.dose,
            context = opt.context,
            data_type = opt.data_type,
            norm_min = opt.norm_min,
            norm_max = opt.norm_max,
        )
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        train_sampler = RandomSampler(
            dataset=train_dataset,
            batch_size=opt.batch_size,
            num_iter=opt.max_iter,
            restore_iter=opt.resume_iter
        )

        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=opt.batch_size,
            sampler=train_sampler,
            shuffle=False,
            drop_last=False,
            num_workers=opt.num_workers,
            pin_memory=True
        )
        self.train_loader = train_loader
        
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=opt.test_batch_size,
            shuffle=False,
            num_workers=opt.num_workers,
            pin_memory=True
        )
        self.test_loader = test_loader

        test_images = [test_dataset[i] for i in range(0, min(300, len(test_dataset)), 75)]
        low_dose = torch.stack([torch.from_numpy(x[0]) for x in test_images], dim=0).cuda()
        full_dose = torch.stack([torch.from_numpy(x[1]) for x in test_images], dim=0).cuda()
        self.test_images = (low_dose, full_dose)

    def fit(self):
        opt = self.opt
        if opt.mode == 'train':
            if opt.resume_iter > 0:
                self.logger.load_checkpoints(opt.resume_iter)

            # training routine
            loader = iter(self.train_loader)
            for n_iter in tqdm.trange(opt.resume_iter + 1, opt.max_iter + 1, disable=(self.rank != 0)):
                inputs = next(loader)
                self.train(inputs, n_iter)
                if n_iter % opt.save_freq == 0:
                    self.logger.checkpoints(n_iter)
                    self.test(n_iter)
                    self.generate_images(n_iter)
            wandb.finish()

        elif opt.mode == 'test':
            self.logger.load_checkpoints(opt.test_iter)
            self.test(opt.test_iter, up_log=False, save_data=True)
            self.generate_images(opt.test_iter)

    def set_model(opt):
        pass

    def train(self, inputs, n_iter):
        pass

    @torch.no_grad()
    def test(self, n_iter):
        pass

    @torch.no_grad()
    def generate_images(self, n_iter, mode):
        pass

    def adjust_learning_rate(self, n_iter):
        opt = self.opt
        pass

    def recon(self, yi, radon, u_water=0.0192, kappa=1.5):
        yi[yi <= 0] = 1
        li_hat = -torch.log(yi)
        sino = li_hat * kappa
        filtered_sino = radon.filter_sinogram(sino, "ramp")
        recon_img = radon.backprojection(filtered_sino)
        recon_img = (recon_img - u_water) / u_water * 1000
        recon_img = (recon_img + 1024) / (3072 + 1024)
        return recon_img

    def transfer_HU(self, img, MIN_B=-1024, MAX_B=3072):
        img = img * (MAX_B - MIN_B) + MIN_B
        return img.short()

    def transfer_calculate_window(self, img, MIN_B=-1024, MAX_B=3072, cut_min=-1000, cut_max=1000):
        img = img * (MAX_B - MIN_B) + MIN_B
        img[img < cut_min] = cut_min
        img[img > cut_max] = cut_max
        img = 255 * (img - cut_min) / (cut_max - cut_min)
        return img

    def transfer_display_window(self, img, MIN_B=-1024, MAX_B=3072, min_v=-160, max_v=240):
        img = img * (MAX_B - MIN_B) + MIN_B
        img = (img - min_v) / (max_v - min_v)
        return img
