CUDA_VISIBLE_DEVICES=gpu_id python main.py \
  --mode train\
  --dataset mayo2016 \
  --dose 0 \
  --image_size 256 \
  --sampling_steps 1000
