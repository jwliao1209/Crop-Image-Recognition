#!/bin/bash

device=4,5
CUDA_VISIBLE_DEVICES=$device python ../scripts/infer.py -c $1