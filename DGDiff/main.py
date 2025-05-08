import argparse
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer


def main(args):
    folder = 'your data root'

    model = Unet(
        channels=1,
        dim=64,
        dim_mults=(1, 2, 4, 8),
        flash_attn=False,
        use_cond=args.use_cond
    )
    diffusion = GaussianDiffusion(
        model,
        image_size=args.image_size,
        timesteps=1000,
        sampling_timesteps=args.sampling_steps,
        ddim_sampling_eta=0.0,
        sampler_=args.sampler_
    )

    trainer = Trainer(
        diffusion,
        folder,
        train_batch_size=args.train_batch_size,
        train_lr=args.train_lr,
        train_num_steps=700000,  # total training steps
        # gradient_accumulate_every=int(64//args.train_batch_size),#2,  # gradient accumulation steps
        gradient_accumulate_every=2,
        ema_decay=0.995,  # exponential moving average decay
        save_and_sample_every=args.save_freq,
        results_folder=args.results_folder,
        amp=True,  # turn on mixed precision
        calculate_fid=False,  # whether to calculate fid during training
        use_cond=args.use_cond,
        image_size=args.image_size,
        cond_size=args.cond_size,
        resume_step=args.resume_step
    )

    if args.mode == 'train':
        trainer.train()
    elif args.mode == 'test':
        trainer.test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Default arguments for training of different methods')
    parser.add_argument("--mode", type=str, default='test')
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument('--dataset', type=str, default='train')

    # train
    parser.add_argument('--results_folder', type=str, default='default')
    parser.add_argument('--train_batch_size', type=int, default=2,
                        help='test_batch_size')
    parser.add_argument('--resume_step', type=int, default=0)
    parser.add_argument('--train_lr', type=float, default=8e-5)
    parser.add_argument('--save_freq', type=int, default=10000,
                        help='save frequency')

    # test
    parser.add_argument('--test_batch_size', type=int, default=1,
                        help='test_batch_size')
    parser.add_argument('--test_epoch', type=int, default=70,
                        help='number of epochs for test')

    parser.add_argument('--dose', type=int, default=2)
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--cond_size', type=int, default=256)
    parser.add_argument('--sampling_steps', type=int, default=30)
    parser.add_argument('--use_cond', action='store_true')
    parser.add_argument('--sampler_', type=str, default='ddim',
                       help='ddim or dpm_solver')

    args = parser.parse_args()
    main(args)
