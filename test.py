# -*- coding=utf-8 -*-
"""
基于风格解耦的深度换脸视频检测算法(Image Style Decoupling Network, ISDN)
2023.04.05，ljc
"""

import sys
sys.setrecursionlimit(15000)    # 手工设置递归调用深度为15000
import os
from tqdm import tqdm           # 显示进度条
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import csv
from skimage import io, transform
import re
import numpy as np
import pandas as pd
import json
import time
import cv2
import math
import albumentations as A

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as T
from torch.utils.data.sampler import RandomSampler
from torch import nn, optim

from data.dataset import RGB_Face_Dataset
from src.net.net import Two_Stream_Net
from src.utils import restore_model, printParams, check_dataPath
from src.utils import test_model
import src.params as params


import warnings
warnings.filterwarnings("ignore")
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def seed_torch(seed):
    """固定随机种子，保证每次测试结果是一样的"""
    # random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True       # 确定性估计
    torch.backends.cudnn.benchmark = True           # 设置False会确定性地选择算法，会降低性能
    torch.backends.cudnn.enabled = True             # 增加运行效率，默认就是True


def test_Illumination():
    """
    模型测试，其中模型为基于FF++数据集内的多域数据训练的ISDN_R网络
    输入：input_face, input_mask, input_label, input_Domain_ID
    """
    seed_torch(seed=2023)

    if torch.cuda.is_available():
        print("GPU is available")  # 判断能否使用GPU
        torch.cuda.set_device(params.gpu_id)  # 指定使用第几块GPU
        print("model location: " + str(params.gpu_id))
    model = restore_model(net=Two_Stream_Net(xcep=False), path=params.savePath_Illumination)
    for i in range(6):
        """加载数据集"""
        
        if i == 0:
            # DFD(c23)
            params.test_dataPath_T = params.DFD_dataPath_c23_test
            params.test_dataPath_F = params.DFD_dataPath_c23_test
        elif i == 1:
            # DFDC
            params.test_dataPath_T = params.DFDC_dataPath_real_test
            params.test_dataPath_F = params.DFDC_dataPath_fake_test
        elif i== 2:
            # DFDC-P
            params.test_dataPath_T = params.DFDC_P_dataPath_real_ori_test
            params.test_dataPath_F = params.DFDC_P_dataPath_fake_ori_test
        elif i == 3:
            # CDF_v1
            params.test_dataPath_T = params.CDF_v1_dataPath_test
            params.test_dataPath_F = params.CDF_v1_dataPath_test
        elif i == 4:
            # CDF_v2
            params.test_dataPath_T = params.CDF_v2_dataPath_real_test
            params.test_dataPath_F = params.CDF_v2_dataPath_fake_test
        elif i == 5:
            # Deeper
            params.test_dataPath_T = params.Deeper_dataPath_test_new
            params.test_dataPath_F = params.Deeper_dataPath_test_new
        elif i == 6:
            params.test_dataPath_T = params.FF_dataPath_real_c23_test
            params.test_dataPath_F = params.FF_dataPath_FSh_c23_test
        
        test_data_T = RGB_Face_Dataset(dataPath=params.test_dataPath_T, transform=True, sampleType='real', maskType='face',
                                        faceRotate=False, mode_aug=False, image_size=256, mask_size=256, domain_ID=0)
        test_data_F = RGB_Face_Dataset(dataPath=params.test_dataPath_F, transform=True, sampleType='fake', maskType='face',
                                        faceRotate=False, mode_aug=False, image_size=256, mask_size=256, domain_ID=1)
        
        sampler_test_data_T = RandomSampler(data_source=test_data_T, num_samples=params.test_num_samples, replacement=True)
        sampler_test_data_F = RandomSampler(data_source=test_data_F, num_samples=params.test_num_samples, replacement=True)

        test_D1_T_dataloader = DataLoader(test_data_T, batch_size=params.sample_size, shuffle=False,
                                        sampler=sampler_test_data_T, num_workers=4, pin_memory=True)
        test_D1_F_dataloader = DataLoader(test_data_F, batch_size=params.sample_size, shuffle=False,
                                        sampler=sampler_test_data_F, num_workers=4, pin_memory=True)
        print('\ntest_data_source1 [sum: T={}, F={}]: {}'.format(len(test_D1_T_dataloader), len(test_D1_F_dataloader), params.test_dataPath_F))


        """模型测试"""
        test_dataloader_list = [test_D1_T_dataloader, test_D1_F_dataloader]
        print("======== 【test】 ========")
        test_model(model, test_dataloader_list)


if __name__ == '__main__':

    # 测试模型
   test_Illumination()

