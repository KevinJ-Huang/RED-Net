#!/usr/bin/env python
# -*-coding:utf-8 -*-

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

class PSNR(nn.Module):
    def __init__(self):
        super(PSNR, self).__init__()

    def forward(self, output, target):
        # diff = np.abs(output.data.cpu().numpy() - target.data.cpu().numpy()) ** 2
        # print (output.data.shape)
        # rmse = np.sqrt(diff.sum() / (output.data.shape[2] * output.data.shape[3]))
        rmse = np.sqrt(np.mean((output.data.cpu().numpy() - target.data.cpu().numpy())**2))
        PIXEL_MAX = 255.0
        psnr = 20 * np.log10(PIXEL_MAX / rmse)
        return psnr
