#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python train.py \
    --resume=weights/multi_step2/ckpt_500.pth \
    --dataset=ICDAR_2015 \
    --batch_size=8 \
    --multi_scale=False \
    --logdir=logs/multi_step2/ \
    --save_folder=weights/multi_step2/ \
    --num_workers=6
