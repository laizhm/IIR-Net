# -*- coding=utf-8 -*-

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
from torchnet import meter
from torchsummary import summary
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, mean_absolute_error, classification_report
from sklearn.metrics import roc_curve, roc_auc_score, auc
from torch.utils.data.sampler import WeightedRandomSampler, RandomSampler
from torch import nn, optim

from src.datasets.datasets import RGB_Face_Dataset
from src.net.net import Two_Stream_Net
from src.utils import restore_model, printParams, check_dataPath
from src.utils import train_model
import src.params as params

os.environ['TMPDIR'] = '/media/tmp_lzm'
os.makedirs('/media/tmp_lzm', exist_ok=True)
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


def train_Illumination():
    """
    基于FF++数据集内的多域数据，训练Illumination_Net网络
    将真样本数据集划分为不重叠的4份
    输入：input_face, input_mask, input_label, input_Domain_ID
    """
    if torch.cuda.is_available():
        print("GPU is available")  # 判断能否使用GPU
        torch.cuda.set_device(params.gpu_id)  # 指定使用第几块GPU
        print("model location: " + str(params.gpu_id))

    """加载数据集"""
    # 1.3跨伪造方法
    params.train_dataPath_D1_T = params.FF_dataPath_real_c23_train_gap5_sub1
    params.train_dataPath_D1_F = params.FF_dataPath_DF_c23_train_gap5
    params.train_dataPath_D2_T = params.FF_dataPath_real_c23_train_gap5_sub2
    params.train_dataPath_D2_F = params.FF_dataPath_FS_c23_train_gap5
    params.train_dataPath_D3_T = params.FF_dataPath_real_c23_train_gap5_sub3
    params.train_dataPath_D3_F = params.FF_dataPath_F2F_c23_train_gap5
    params.train_dataPath_D4_T = params.FF_dataPath_real_c23_train_gap5_sub4
    params.train_dataPath_D4_F = params.FF_dataPath_NT_c23_train_gap5

    params.valid_dataPath_T = params.DFD_dataPath_c23_test
    params.valid_dataPath_F = params.DFD_dataPath_c23_test
    params.valid_dataPath_T2 = params.CDF_v2_dataPath_real_test
    params.valid_dataPath_F2 = params.CDF_v2_dataPath_fake_test
    params.valid_dataPath_T3 = params.DFDC_dataPath_real_test
    params.valid_dataPath_F3 = params.DFDC_dataPath_fake_test
    params.valid_dataPath_T4 = params.Deeper_dataPath_test_new
    params.valid_dataPath_F4 = params.Deeper_dataPath_test_new
    params.valid_dataPath_T5 = params.DFDC_P_dataPath_real_ori_test
    params.valid_dataPath_F5 = params.DFDC_P_dataPath_fake_ori_test
    

    # 根据使用的服务器来检查和修改说使用的txt文件。因为不同服务器的数据存放在不同位置，需要根据使用的服务器来自动修改数据的路径
    params.train_dataPath_D1_T = check_dataPath(dataPath=params.train_dataPath_D1_T)
    params.train_dataPath_D1_F = check_dataPath(dataPath=params.train_dataPath_D1_F)
    params.train_dataPath_D2_T = check_dataPath(dataPath=params.train_dataPath_D2_T)
    params.train_dataPath_D2_F = check_dataPath(dataPath=params.train_dataPath_D2_F)
    params.train_dataPath_D3_T = check_dataPath(dataPath=params.train_dataPath_D3_T)
    params.train_dataPath_D3_F = check_dataPath(dataPath=params.train_dataPath_D3_F)
    params.train_dataPath_D4_T = check_dataPath(dataPath=params.train_dataPath_D4_T)
    params.train_dataPath_D4_F = check_dataPath(dataPath=params.train_dataPath_D4_F)
    params.valid_dataPath_T = check_dataPath(dataPath=params.valid_dataPath_T)
    params.valid_dataPath_F = check_dataPath(dataPath=params.valid_dataPath_F)
    params.valid_dataPath_T2 = check_dataPath(dataPath=params.valid_dataPath_T2)
    params.valid_dataPath_F2 = check_dataPath(dataPath=params.valid_dataPath_F2)
    params.valid_dataPath_T3 = check_dataPath(dataPath=params.valid_dataPath_T3)
    params.valid_dataPath_F3 = check_dataPath(dataPath=params.valid_dataPath_F3)
    params.valid_dataPath_T4 = check_dataPath(dataPath=params.valid_dataPath_T4)
    params.valid_dataPath_F4 = check_dataPath(dataPath=params.valid_dataPath_F4)
    params.valid_dataPath_T5 = check_dataPath(dataPath=params.valid_dataPath_T5)
    params.valid_dataPath_F5 = check_dataPath(dataPath=params.valid_dataPath_F5)
    # 打印参数
    printParams()

    train_D1_data_T = RGB_Face_Dataset(dataPath=params.train_dataPath_D1_T, transform=True, sampleType='real',
                                       maskType='face',
                                       faceRotate=False, mode_aug=True, image_size=256, mask_size=256, domain_ID=0)
    train_D2_data_T = RGB_Face_Dataset(dataPath=params.train_dataPath_D2_T, transform=True, sampleType='real',
                                       maskType='face',
                                       faceRotate=False, mode_aug=True, image_size=256, mask_size=256, domain_ID=0)
    train_D3_data_T = RGB_Face_Dataset(dataPath=params.train_dataPath_D3_T, transform=True, sampleType='real',
                                       maskType='face',
                                       faceRotate=False, mode_aug=True, image_size=256, mask_size=256, domain_ID=0)
    train_D4_data_T = RGB_Face_Dataset(dataPath=params.train_dataPath_D4_T, transform=True, sampleType='real',
                                       maskType='face',
                                       faceRotate=False, mode_aug=True, image_size=256, mask_size=256, domain_ID=0)
    
    train_D1_data_F = RGB_Face_Dataset(dataPath=params.train_dataPath_D1_F, transform=True, sampleType='fake',
                                       maskType='face',
                                       faceRotate=False, mode_aug=True, image_size=256, mask_size=256, domain_ID=1)
    train_D2_data_F = RGB_Face_Dataset(dataPath=params.train_dataPath_D2_F, transform=True, sampleType='fake',
                                       maskType='face',
                                       faceRotate=False, mode_aug=True, image_size=256, mask_size=256, domain_ID=2)
    train_D3_data_F = RGB_Face_Dataset(dataPath=params.train_dataPath_D3_F, transform=True, sampleType='fake',
                                       maskType='face',
                                       faceRotate=False, mode_aug=True, image_size=256, mask_size=256, domain_ID=3)
    train_D4_data_F = RGB_Face_Dataset(dataPath=params.train_dataPath_D4_F, transform=True, sampleType='fake',
                                       maskType='face',
                                       faceRotate=False, mode_aug=True, image_size=256, mask_size=256, domain_ID=4)

    train_D1_sbi_F = RGB_Face_Dataset(dataPath=params.train_dataPath_D1_T, transform=True, sampleType='real',
                                       maskType='face',
                                       faceRotate=False, mode_aug=True, image_size=256, mask_size=256, domain_ID=5, model_sbi=True, sbi_aug=False)
    train_D2_sbi_F = RGB_Face_Dataset(dataPath=params.train_dataPath_D2_T, transform=True, sampleType='real',
                                       maskType='face',
                                       faceRotate=False, mode_aug=True, image_size=256, mask_size=256, domain_ID=5, model_sbi=True, sbi_aug=False)
    train_D3_sbi_F = RGB_Face_Dataset(dataPath=params.train_dataPath_D3_T, transform=True, sampleType='real',
                                       maskType='face',
                                       faceRotate=False, mode_aug=True, image_size=256, mask_size=256, domain_ID=5, model_sbi=True, sbi_aug=False)
    train_D4_sbi_F = RGB_Face_Dataset(dataPath=params.train_dataPath_D4_T, transform=True, sampleType='real',
                                       maskType='face',
                                       faceRotate=False, mode_aug=True, image_size=256, mask_size=256, domain_ID=5, model_sbi=True, sbi_aug=False)
    
    train_D1_sbi_T = RGB_Face_Dataset(dataPath=params.train_dataPath_D1_T, transform=True, sampleType='real',
                                       maskType='face',
                                       faceRotate=False, mode_aug=True, image_size=256, mask_size=256, domain_ID=5, model_sbi=True, sbi_aug=True)
    train_D2_sbi_T = RGB_Face_Dataset(dataPath=params.train_dataPath_D2_T, transform=True, sampleType='real',
                                       maskType='face',
                                       faceRotate=False, mode_aug=True, image_size=256, mask_size=256, domain_ID=5, model_sbi=True, sbi_aug=True)
    train_D3_sbi_T = RGB_Face_Dataset(dataPath=params.train_dataPath_D3_T, transform=True, sampleType='real',
                                       maskType='face',
                                       faceRotate=False, mode_aug=True, image_size=256, mask_size=256, domain_ID=5, model_sbi=True, sbi_aug=True)
    train_D4_sbi_T = RGB_Face_Dataset(dataPath=params.train_dataPath_D4_T, transform=True, sampleType='real',
                                       maskType='face',
                                       faceRotate=False, mode_aug=True, image_size=256, mask_size=256, domain_ID=5, model_sbi=True, sbi_aug=True)
    
    sampler_train_D1_data_T = RandomSampler(data_source=train_D1_data_T, num_samples=params.train_num_samples,
                                            replacement=True)
    sampler_train_D1_data_F = RandomSampler(data_source=train_D1_data_F, num_samples=params.train_num_samples,
                                            replacement=True)
    sampler_train_D2_data_T = RandomSampler(data_source=train_D2_data_T, num_samples=params.train_num_samples,
                                            replacement=True)
    sampler_train_D2_data_F = RandomSampler(data_source=train_D2_data_F, num_samples=params.train_num_samples,
                                            replacement=True)
    sampler_train_D3_data_T = RandomSampler(data_source=train_D3_data_T, num_samples=params.train_num_samples,
                                            replacement=True)
    sampler_train_D3_data_F = RandomSampler(data_source=train_D3_data_F, num_samples=params.train_num_samples,
                                            replacement=True)
    sampler_train_D4_data_T = RandomSampler(data_source=train_D4_data_T, num_samples=params.train_num_samples,
                                            replacement=True)
    sampler_train_D4_data_F = RandomSampler(data_source=train_D4_data_F, num_samples=params.train_num_samples,
                                            replacement=True)
    
    sampler_train_D1_sbi_T = RandomSampler(data_source=train_D1_sbi_T, num_samples=params.train_num_samples,
                                            replacement=True)
    sampler_train_D1_sbi_F = RandomSampler(data_source=train_D1_sbi_F, num_samples=params.train_num_samples,
                                            replacement=True)
    sampler_train_D2_sbi_T = RandomSampler(data_source=train_D2_sbi_T, num_samples=params.train_num_samples,
                                            replacement=True)
    sampler_train_D2_sbi_F = RandomSampler(data_source=train_D2_sbi_F, num_samples=params.train_num_samples,
                                            replacement=True)
    sampler_train_D3_sbi_T = RandomSampler(data_source=train_D3_sbi_T, num_samples=params.train_num_samples,
                                            replacement=True)
    sampler_train_D3_sbi_F = RandomSampler(data_source=train_D3_sbi_F, num_samples=params.train_num_samples,
                                            replacement=True)
    sampler_train_D4_sbi_T = RandomSampler(data_source=train_D4_sbi_T, num_samples=params.train_num_samples,
                                            replacement=True)
    sampler_train_D4_sbi_F = RandomSampler(data_source=train_D4_sbi_F, num_samples=params.train_num_samples,
                                            replacement=True)
    train_D1_T_dataloader = DataLoader(train_D1_data_T, batch_size=params.sample_size, shuffle=False,
                                       sampler=sampler_train_D1_data_T, num_workers=params.num_workers, pin_memory=True)
    train_D1_F_dataloader = DataLoader(train_D1_data_F, batch_size=params.sample_size, shuffle=False,
                                       sampler=sampler_train_D1_data_F, num_workers=params.num_workers, pin_memory=True)
    train_D2_T_dataloader = DataLoader(train_D2_data_T, batch_size=params.sample_size, shuffle=False,
                                       sampler=sampler_train_D2_data_T, num_workers=params.num_workers, pin_memory=True)
    train_D2_F_dataloader = DataLoader(train_D2_data_F, batch_size=params.sample_size, shuffle=False,
                                       sampler=sampler_train_D2_data_F, num_workers=params.num_workers, pin_memory=True)
    train_D3_T_dataloader = DataLoader(train_D3_data_T, batch_size=params.sample_size, shuffle=False,
                                       sampler=sampler_train_D3_data_T, num_workers=params.num_workers, pin_memory=True)
    train_D3_F_dataloader = DataLoader(train_D3_data_F, batch_size=params.sample_size, shuffle=False,
                                       sampler=sampler_train_D3_data_F, num_workers=params.num_workers, pin_memory=True)
    train_D4_T_dataloader = DataLoader(train_D4_data_T, batch_size=params.sample_size, shuffle=False,
                                       sampler=sampler_train_D4_data_T, num_workers=params.num_workers, pin_memory=True)
    train_D4_F_dataloader = DataLoader(train_D4_data_F, batch_size=params.sample_size, shuffle=False,
                                       sampler=sampler_train_D4_data_F, num_workers=params.num_workers, pin_memory=True)
    
    train_D1_T_sbi_dataloader = DataLoader(train_D1_sbi_T, batch_size=params.sample_size, shuffle=False,
                                       sampler=sampler_train_D1_sbi_T, num_workers=params.num_workers, pin_memory=True)
    train_D1_F_sbi_dataloader = DataLoader(train_D1_sbi_F, batch_size=params.sample_size, shuffle=False,
                                       sampler=sampler_train_D1_sbi_F, num_workers=params.num_workers, pin_memory=True)
    train_D2_T_sbi_dataloader = DataLoader(train_D2_sbi_T, batch_size=params.sample_size, shuffle=False,
                                       sampler=sampler_train_D2_sbi_T, num_workers=params.num_workers, pin_memory=True)
    train_D2_F_sbi_dataloader = DataLoader(train_D2_sbi_F, batch_size=params.sample_size, shuffle=False,
                                       sampler=sampler_train_D2_sbi_F, num_workers=params.num_workers, pin_memory=True)
    train_D3_T_sbi_dataloader = DataLoader(train_D3_sbi_T, batch_size=params.sample_size, shuffle=False,
                                       sampler=sampler_train_D3_sbi_T, num_workers=params.num_workers, pin_memory=True)
    train_D3_F_sbi_dataloader = DataLoader(train_D3_sbi_F, batch_size=params.sample_size, shuffle=False,
                                       sampler=sampler_train_D3_sbi_F, num_workers=params.num_workers, pin_memory=True)
    train_D4_T_sbi_dataloader = DataLoader(train_D4_sbi_T, batch_size=params.sample_size, shuffle=False,
                                       sampler=sampler_train_D4_sbi_T, num_workers=params.num_workers, pin_memory=True)
    train_D4_F_sbi_dataloader = DataLoader(train_D4_sbi_F, batch_size=params.sample_size, shuffle=False,
                                       sampler=sampler_train_D4_sbi_F, num_workers=params.num_workers, pin_memory=True)
    """    
    dataloader为迭代器，每个epoch循环时，dataloader会自动利用sampler采样不同的num_samples个样本，然后将num_samples个样本分为若干个batch
    这里的len(dataloader)表示的是sampler采样的num_samples个样本，可以被分为多少个batch_size，即num_samples=len(dataloader)*batch_size
    """

    # 验证集数据，要注意：验证时不要使用数据增强，即mode_aug=False
    valid_data_T = RGB_Face_Dataset(dataPath=params.valid_dataPath_T, transform=True, sampleType='real',
                                    maskType='face',
                                    faceRotate=False, mode_aug=False, image_size=256, mask_size=256, domain_ID=5)
    valid_data_F = RGB_Face_Dataset(dataPath=params.valid_dataPath_F, transform=True, sampleType='fake',
                                    maskType='face',
                                    faceRotate=False, mode_aug=False, image_size=256, mask_size=256, domain_ID=5)
    valid_data_T2 = RGB_Face_Dataset(dataPath=params.valid_dataPath_T2, transform=True, sampleType='real',
                                    maskType='face',
                                    faceRotate=False, mode_aug=False, image_size=256, mask_size=256, domain_ID=5)
    valid_data_F2 = RGB_Face_Dataset(dataPath=params.valid_dataPath_F2, transform=True, sampleType='fake',
                                    maskType='face',
                                    faceRotate=False, mode_aug=False, image_size=256, mask_size=256, domain_ID=5)
    valid_data_T3 = RGB_Face_Dataset(dataPath=params.valid_dataPath_T3, transform=True, sampleType='real',
                                    maskType='face',
                                    faceRotate=False, mode_aug=False, image_size=256, mask_size=256, domain_ID=5)
    valid_data_F3 = RGB_Face_Dataset(dataPath=params.valid_dataPath_F3, transform=True, sampleType='fake',
                                    maskType='face',
                                    faceRotate=False, mode_aug=False, image_size=256, mask_size=256, domain_ID=5)
    valid_data_T4 = RGB_Face_Dataset(dataPath=params.valid_dataPath_T4, transform=True, sampleType='real',
                                    maskType='face',
                                    faceRotate=False, mode_aug=False, image_size=256, mask_size=256, domain_ID=5)
    valid_data_F4 = RGB_Face_Dataset(dataPath=params.valid_dataPath_F4, transform=True, sampleType='fake',
                                    maskType='face',
                                    faceRotate=False, mode_aug=False, image_size=256, mask_size=256, domain_ID=5)
    valid_data_T5 = RGB_Face_Dataset(dataPath=params.valid_dataPath_T5, transform=True, sampleType='real',
                                    maskType='face',
                                    faceRotate=False, mode_aug=False, image_size=256, mask_size=256, domain_ID=5)
    valid_data_F5 = RGB_Face_Dataset(dataPath=params.valid_dataPath_F5, transform=True, sampleType='fake',
                                    maskType='face',
                                    faceRotate=False, mode_aug=False, image_size=256, mask_size=256, domain_ID=5)
    sampler_valid_data_T = RandomSampler(data_source=valid_data_T, num_samples=params.valid_num_samples,
                                         replacement=True)
    sampler_valid_data_F = RandomSampler(data_source=valid_data_F, num_samples=params.valid_num_samples,
                                         replacement=True)
    sampler_valid_data_T2 = RandomSampler(data_source=valid_data_T2, num_samples=params.valid_num_samples,
                                         replacement=True)
    sampler_valid_data_F2 = RandomSampler(data_source=valid_data_F2, num_samples=params.valid_num_samples,
                                         replacement=True)
    sampler_valid_data_T3 = RandomSampler(data_source=valid_data_T3, num_samples=params.valid_num_samples,
                                         replacement=True)
    sampler_valid_data_F3 = RandomSampler(data_source=valid_data_F3, num_samples=params.valid_num_samples,
                                         replacement=True)
    sampler_valid_data_T4 = RandomSampler(data_source=valid_data_T4, num_samples=params.valid_num_samples,
                                         replacement=True)
    sampler_valid_data_F4 = RandomSampler(data_source=valid_data_F4, num_samples=params.valid_num_samples,
                                         replacement=True)
    sampler_valid_data_T5 = RandomSampler(data_source=valid_data_T5, num_samples=params.valid_num_samples,
                                         replacement=True)
    sampler_valid_data_F5 = RandomSampler(data_source=valid_data_F5, num_samples=params.valid_num_samples,
                                         replacement=True)
    valid_D1_T_dataloader = DataLoader(valid_data_T, batch_size=params.sample_size, shuffle=False,
                                       sampler=sampler_valid_data_T, num_workers=params.num_workers2, pin_memory=True)
    valid_D1_F_dataloader = DataLoader(valid_data_F, batch_size=params.sample_size, shuffle=False,
                                       sampler=sampler_valid_data_F, num_workers=params.num_workers2, pin_memory=True)
    valid_D1_T_dataloader2 = DataLoader(valid_data_T2, batch_size=params.sample_size, shuffle=False,
                                       sampler=sampler_valid_data_T2, num_workers=params.num_workers2, pin_memory=True)
    valid_D1_F_dataloader2 = DataLoader(valid_data_F2, batch_size=params.sample_size, shuffle=False,
                                       sampler=sampler_valid_data_F2, num_workers=params.num_workers2, pin_memory=True)
    valid_D1_T_dataloader3 = DataLoader(valid_data_T3, batch_size=params.sample_size, shuffle=False,
                                       sampler=sampler_valid_data_T3, num_workers=params.num_workers2, pin_memory=True)
    valid_D1_F_dataloader3 = DataLoader(valid_data_F3, batch_size=params.sample_size, shuffle=False,
                                       sampler=sampler_valid_data_F3, num_workers=params.num_workers2, pin_memory=True)
    valid_D1_T_dataloader4 = DataLoader(valid_data_T4, batch_size=params.sample_size, shuffle=False,
                                       sampler=sampler_valid_data_T4, num_workers=params.num_workers2, pin_memory=True)
    valid_D1_F_dataloader4 = DataLoader(valid_data_F4, batch_size=params.sample_size, shuffle=False,
                                       sampler=sampler_valid_data_F4, num_workers=params.num_workers2, pin_memory=True)
    valid_D1_T_dataloader5 = DataLoader(valid_data_T5, batch_size=params.sample_size, shuffle=False,
                                       sampler=sampler_valid_data_T5, num_workers=params.num_workers2, pin_memory=True)
    valid_D1_F_dataloader5 = DataLoader(valid_data_F5, batch_size=params.sample_size, shuffle=False,
                                       sampler=sampler_valid_data_F5, num_workers=params.num_workers2, pin_memory=True)

    # 初始化检测模型
    model = restore_model(net=Two_Stream_Net(xcep=True), path=params.savePath_Illumination)

    print("\n【model location】:" + str(next(model.parameters()).device))

    """模型训练和验证"""
    train_dataloader_list = [train_D1_T_dataloader, train_D1_F_dataloader, train_D2_T_dataloader, train_D2_F_dataloader,
                             train_D3_T_dataloader, train_D3_F_dataloader, train_D4_T_dataloader, train_D4_F_dataloader,
                             train_D1_T_sbi_dataloader, train_D1_F_sbi_dataloader, train_D2_T_sbi_dataloader, train_D2_F_sbi_dataloader,
                             train_D3_T_sbi_dataloader, train_D3_F_sbi_dataloader, train_D4_T_sbi_dataloader, train_D4_F_sbi_dataloader]
    valid_dataloader_list = [valid_D1_T_dataloader, valid_D1_F_dataloader]
    valid_dataloader_list2 = [valid_D1_T_dataloader2, valid_D1_F_dataloader2]
    valid_dataloader_list3 = [valid_D1_T_dataloader3, valid_D1_F_dataloader3]
    valid_dataloader_list4 = [valid_D1_T_dataloader4, valid_D1_F_dataloader4]
    valid_dataloader_list5 = [valid_D1_T_dataloader5, valid_D1_F_dataloader5]
    valid_dataloader_list = [valid_dataloader_list, valid_dataloader_list2, valid_dataloader_list3, valid_dataloader_list4, valid_dataloader_list5]

    print("======== 【train】 ========")
    train_model(model, train_dataloader_list, valid_dataloader_list)


if __name__ == '__main__':

    # 训练模型
    train_Illumination()


