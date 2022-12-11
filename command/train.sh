#!/bin/bash

device=4,5
CUDA_VISIBLE_DEVICES=$device python ../scripts/train.py -c $1
