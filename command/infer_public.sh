#!/bin/bash

device=0
CUDA_VISIBLE_DEVICES=$device python ../scripts/infer_public.py -c $1
