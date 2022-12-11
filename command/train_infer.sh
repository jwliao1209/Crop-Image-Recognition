#!/bin/bash

device=0,1
CUDA_VISIBLE_DEVICES=$device python ../scripts/train.py -c $1

CUDA_VISIBLE_DEVICES=$device python ../scripts/infer.py -c $1
