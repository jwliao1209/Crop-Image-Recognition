#!/bin/bash

python3 train.py --model efficientnet_b0 \
                 --image_size 768 \
                 -ep 100 \
                 -bs 32 \
                 -agbs 1 \
                 --loss FL \
                 --lr 3e-4 \
                 --step_size 2000 \
                 --fold 1 \
                 --device 2 3


# python3 train.py --model swin_s \
#                  --image_size 512 \
#                  -ep 100 \
#                  -bs 24 \
#                  --loss FL \
#                  --lr 8e-5 \
#                  --step_size 5000 \
#                  --fold 1 \
#                  --device 2 3


# python3 train.py --model convnext_s \
#                  --image_size 512 \
#                  -ep 100 \
#                  -bs 40 \
#                  --loss FL \
#                  --lr 3e-4 \
#                  --step_size 5000 \
#                  --fold 1 \
#                  --device 2 3
