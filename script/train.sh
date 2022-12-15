#!/bin/bash

python3 train.py --model efficientnet_b0 \
                 --image_size 1080 \
                 -ep 50 \
                 -bs 10 \
                 -agbs 1 \
                 --trans v1 \
                 --loss FL \
                 --optim SGD \
                 --lr 3e-2 \
                 --scheduler warmup_cos \
                 --fold 0 \
                 --device 5 \
                 --num_workers 50 \
                 --autoaug 1
