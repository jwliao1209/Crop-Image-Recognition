import os
from datetime import datetime
from .utils import get_time


RANDOM_SEED       = 614
BASELINE_F1_SCORE = 0.
TEST_BS           = 1
SAVE_TOPK         = 5
CUR_TIME          = get_time()

DATA_ROOT    = './dataset'
SAVE_ROOT    = './checkpoint'
MONITOR_ROOT = './tensorboard'
PRED_DIR     = './submission'

SAVE_DIR       = os.path.join(SAVE_ROOT, CUR_TIME)    
WEIGHT_DIR     = os.path.join(SAVE_DIR, 'weight')
TENSORBORD_DIR = os.path.join(MONITOR_ROOT, CUR_TIME)
CONFIG_PATH    = os.path.join(SAVE_DIR, 'config.json')
LOG_PATH       = os.path.join(SAVE_DIR, 'result.csv')

LABEL_CATEGORY_MAP = {
    0:  'asparagus',
    1:  'bambooshoots',
    2:  'betel',
    3:  'broccoli',
    4:  'cauliflower',
    5:  'chinesecabbage',
    6:  'chinesechives',
    7:  'custardapple',
    8:  'grape',
    9:  'greenhouse',
    10: 'greenonion',
    11: 'kale',
    12: 'lemon',
    13: 'lettuce',
    14: 'litchi',
    15: 'longan',
    16: 'loofah',
    17: 'mango',
    18: 'onion',
    19: 'others',
    20: 'papaya',
    21: 'passionfruit',
    22: 'pear',
    23: 'pennisetum',
    24: 'redbeans',
    25: 'roseapple',
    26: 'sesbania',
    27: 'soybeans',
    28: 'sunhemp',
    29: 'sweetpotato',
    30: 'taro',
    31: 'tea',
    32: 'waterbamboo'
    }
