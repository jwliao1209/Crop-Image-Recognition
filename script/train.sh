#!/bin/bash

python3 train.py --model convnext_s \
                 --image_size 1080 \
                 -ep 50 \
                 -bs 4 \
                 -agbs 8 \
                 --trans v1 \
                 --loss FL \
                 --optim SGD \
                 --lr 5e-3 \
                 --scheduler warmup_cos \
                 --fold 0 \
                 --device 5 \
                 --num_workers 50 \
                 --autoaug 1
