#!/bin/bash

# python3 train.py --model efficientnet_b0 \
#                  --image_size 1080 \
#                  -ep 100 \
#                  -bs 2 \
#                  -agbs 1 \
#                  --trans v1 \
#                  --loss FL \
#                  --optim SGD \
#                  --lr 3e-2 \
#                  --scheduler warmup_cos \
#                  --step_size 10000 \
#                  --fold 0 \
#                  --device 3 \
#                  --num_workers 5 \
#                  --autoaug 1

python3 train.py --model efficientnet_b0 \
                 --image_size 384 \
                 -ep 100 \
                 -bs 4 \
                 -agbs 1 \
                 --trans v1 \
                 --loss FL \
                 --optim SGD \
                 --lr 3e-2 \
                 --scheduler warmup_cos \
                 --step_size 10000 \
                 --fold 0 \
                 --device 3 \
                 --num_workers 4 \
                 --autoaug 1


# python3 train.py --model swin_s \
#                  --image_size 512 \
#                  -ep 100 \
#                  -bs 2 \
#                  -agbs 1 \
#                  --trans v1 \
#                  --loss FL \
#                  --optim SGD \
#                  --lr 3e-2 \
#                  --scheduler step \
#                  --step_size 10000 \
#                  --fold 0 \
#                  --device 1 \
#                  --num_workers 50 \
#                  --autoaug 1


# python3 train.py --model vit_l \
#                  --image_size 512 \
#                  -ep 100 \
#                  -bs 10 \
#                  -agbs 32 \
#                  --loss FL \
#                  --lr 8e-5 \
#                  --step_size 5000 \
#                  --fold 0 \
#                  --device 0 1 \
#                  --num_workers 50 \
#                  --autoaug 1

# python3 train.py --model swin_v2_b \
#                  --image_size 384 \
#                  -ep 100 \
#                  -bs 32 \
#                  -agbs 1 \
#                  --loss FL \
#                  --lr 8e-5 \
#                  --step_size 5000 \
#                  --fold 0 \
#                  --device 0 1 \
#                  --num_workers 50 \
#                  --autoaug 1


# python3 train.py --model swin_s \
#                  --image_size 512 \
#                  -ep 100 \
#                  -bs 50 \
#                  --loss FL \
#                  --lr 8e-5 \
#                  --step_size 5000 \
#                  --fold 1 \
#                  --device 2 3


# python3 train.py --model convnext_s \
#                  --image_size 1024 \
#                  -ep 100 \
#                  -bs 8 \
#                  -agbs 1 \
#                  --loss CE \
#                  --optim SGD \
#                  --scheduler reduce \
#                  --lr 7e-4 \
#                  --step_size 2500 \
#                  --fold 0 \
#                  --device 2 3 \
#                  --num_workers 8 \
#                  --autoaug 1