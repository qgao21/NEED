#!/bin/bash

# stage 1: image size 256
CUDA_VISIBLE_DEVICES=gpu_id python main.py \
  --mode test\
  --dataset mayo2016 \
  --dose 2 \
  --image_size 256 \
  --sampling_steps 1000

# stage 2: image size 256 -> 512
CUDA_VISIBLE_DEVICES=gpu_id python main.py \
  --mode test\
  --dataset mayo2016 \
  --dose 2 \
  --cond_size 256 \
  --image_size 512 \
  --sampling_timesteps 30 \
  --use_cond
