CUDA_VISIBLE_DEVICES=gpu_id python main.py \
  --mode train\
  --train_batch_size 2\
  --dataset mayo2016 \
  --dose 0 \
  --image_size 512 \
  --cond_size 256
  --sampling_steps 30\
  --use_cond
