#!/bin/bash

CUDA_VISIBLE_DEVICES=4 python inference.py \
                       --checkpoint 10-01-13-29-04 10-28-11-15-57_size1080 \
                       --topk 1 1
