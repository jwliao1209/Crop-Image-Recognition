#!/bin/bash
model=EfficientB4

python train.py \
--folder 0 \
--batch_size 64 \
--num_class 33 \
--model $model \
--img_size 384 \
--loss CE \
--optim AdamW \
--lr 1e-4 \
--scheduler step \
--num_workers 16 \
--device 0 1 2 3 \
