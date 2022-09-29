#!/bin/bash
model=DenseNet121
# EfficientB4

python train.py \
--folder 0 \
--batch_size 10 \
--num_class 33 \
--train_num 8000 \
--valid_num 3000 \
--model $model \
--img_size 224 \
--loss CE \
--optim AdamW \
--lr 1e-4 \
--weight_decay 1e-4 \
--scheduler cos \
--num_workers 4 \
--device 0 \
