#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import sys
sys.path.insert(0, '/root/hj9/RED-Net/')

# =================== config for train flow ============================================================================

DESC = "RED20 for denosing"
experiment_name = os.path.splitext(__file__.split('/')[-1])[0]

BASE_ROOT = '/root/hj9/RED-Net/'  # server

TRAIN_ROOT      = '/root/hj9/RED-Net/train_task/'+experiment_name

MODEL_FOLDER        = os.path.join(TRAIN_ROOT, 'models')
TRAIN_OUT_FOLDER    = os.path.join(TRAIN_ROOT, 'train_out')
PEEK_OUT_FOLDER     = os.path.join(TRAIN_ROOT, 'peek_out')
TEST_OUT_FOLDER     = os.path.join(TRAIN_ROOT, 'test_out')

DATASET_DIR = '/root/hj9/images2/'
DATASET_ID = 'dataset20171030_gauss_noise03_random_ds_random_kernel_jpeg'
DATASET_TXT_DIR = '/root/hj9/images2/'

IMAGE_SITE_URL      = 'http://172.18.34.6:8099//image-site/dataset/{dataset_name}?page=1&size=50'
IMAGE_SITE_DATA_DIR = '/root/hj9/images2/'

peek_images = ['/root/ca/denoising/BSDS300/experiments/noise/test/38092_10.jpg',  '/root/ca/denoising/BSDS300/experiments/noise/test/42049_10.jpg']
test_input_dir = "/root/hj9/images2/"


GPU_ID = 0
epochs = 1
batch_size = 32
stride = 160
extension = 16
start_epoch = 1
save_snapshot_interval_epoch = 1
peek_interval_epoch = 10000000
save_train_hr_interval_epoch = 1
loss_average_win_size = 5
validate_interval_epoch = 1
plot_loss_start_epoch = 1
only_validate = False  #



from visdom import Visdom
vis = Visdom(server='http://localhost', port=8097)
# vis = None

# =================== net and model =====================================================================
import torch
import torch.nn as nn
from squid.net import Red20ResiDownsample
from squid.model  import SuperviseModel
from squid.metric import PSNR
from squid.data import RandomCropPhoto2PhotoData

image_c = 1
feature_num = 64
target_net = Red20ResiDownsample(image_c=image_c, feature_num=feature_num)

model = SuperviseModel({
    'net': target_net,
    # 'optimizer': torch.optim.Adam([{'name':'net_params', 'params':target_net.parameters(), 'base_lr':1e-4,'warm_epoch':None,'total_epoch':epochs}], betas=(0.9, 0.999), weight_decay=0.0005),
    'optimizer': torch.optim.SGD([{'name':'net_params', 'params':target_net.parameters(), 'base_lr':1e-8,
                                   'warm_epoch':None,'total_epoch':epochs}],lr=1e-8, momentum=0.9, weight_decay=0.0005),
    # 'lr_step_ratio': 10,
    # 'lr_step_size': 1,

    'supervise':{
        'out':{'L2_loss': {'obj': nn.MSELoss(size_average=True),  'factor':1.0, 'weight': 1.0}}
    },
    'metrics':{
        'out': {
            'psnr': {'obj': PSNR()}
        }
    },
    'not_show_gradient':True
})

# =================== dataset =====================================================================
train_dataset = RandomCropPhoto2PhotoData({
            'crop_size': 50,
            'crop_stride': 2,
            'data_root': DATASET_DIR,
            'desc_file_path': os.path.join(DATASET_TXT_DIR, 'train.txt'),
            'is_rotated':True
})

valid_dataset = RandomCropPhoto2PhotoData({
            'crop_size': 50,
            'crop_stride': 2,
            'data_root': DATASET_DIR,
            'desc_file_path': os.path.join(DATASET_TXT_DIR, 'val.txt'),
            'is_rotated':True
})

