#!/bin/bash

device=7
CUDA_VISIBLE_DEVICES=$device python ../scripts/infer_tta.py -c $1