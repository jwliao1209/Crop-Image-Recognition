#!/bin/bash

CUDA_VISIBLE_DEVICES=3 python inference.py \
                       --checkpoint 11-20-16-14-32 \
                       --topk 3
