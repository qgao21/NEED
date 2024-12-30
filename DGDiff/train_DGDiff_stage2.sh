#!/bin/bash

# 第一次运行 test.py
CUDA_VISIBLE_DEVICES=3 python test.py \
  --mode test\
  --test_dataset mayo2016 \
  --dose 2 \
  --image_size 256 \
  --sampling_timesteps 1000

# 第二次运行 test.py
CUDA_VISIBLE_DEVICES=3 python test.py \
  --mode test\
  --test_dataset mayo2016 \
  --dose 4 \
  --cond_size 256 \
  --image_size 512 \
  --sampling_timesteps 30 \
  --use_cond