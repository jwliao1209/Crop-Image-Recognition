#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python inference.py --checkpoint 11-23-11-31-04 --topk 3
CUDA_VISIBLE_DEVICES=0 python inference.py --checkpoint 11-25-19-31-53 --topk 3
CUDA_VISIBLE_DEVICES=0 python inference.py --checkpoint 12-03-14-52-49 --topk 3
CUDA_VISIBLE_DEVICES=0 python inference.py --checkpoint 12-06-10-30-56 --topk 3
