# -*- coding=utf-8 -*-
import sys
from torch.cuda import device
from triton.language import dtype

sys.setrecursionlimit(15000)  # 手工设置递归调用深度为15000
import os
from tqdm import tqdm  # 显示进度条
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
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as T
from torchnet import meter
from torchsummary import summary
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score, auc
from torch.utils.data.sampler import WeightedRandomSampler, RandomSampler
from torch import nn, optim
from torch.backends import cudnn

import src.params as params
from src.net.net import ContrastLoss, ContrastLoss2
from models.Relighting import RelightNet

from sklearn import manifold
import imageio

"""基础功能函数==========================================================================================="""


def restore_model(net, path):
    """加载保存的模型"""

    # 权重初始化
    # net.apply(init_weights)
    if torch.cuda.is_available():  # 判断是否可使用GPU加速
        device = torch.device("cuda:0")
        net = net.to(device)
        net = nn.DataParallel(net)

    if path is not None and os.path.exists(path):  # 判断文件是否存在
        net.load_state_dict(torch.load(path))  # 加载保存的网络参数
        net.restored = True
        # net.restored = False
        print("Restore saved model: {}".format(path))  # 表示模型已保存
    else:
        net.restored = False
        print("Can't Restore saved model: {}".format(path))  # 表示模型未保存

    return net


def save_model(net, root, filename):
    """保存训练后的模型"""
    model_root = root
    if not os.path.exists(model_root):
        os.makedirs(model_root)  # 判断目录是否存在，不存在则递归创建目录

    # 获取当前torch的版本，输出格式为‘1.8.0+cu111’或者‘1.0.1’
    torch_version = str(torch.__version__)
    tv_tmp = torch_version.split('.')  # ['1','8','0+cu111']
    tv = int(tv_tmp[0] + tv_tmp[1])  # '18'->18

    if tv > 16:
        # 仅保存模型的参数，1.6以上的torch使用(3090服务器)，"_use_new_zipfile_serialization=False"可以使得3090和509模型通用
        torch.save(net.state_dict(), os.path.join(model_root, filename), _use_new_zipfile_serialization=False)
    else:
        # 仅保存模型的参数，1.6以下的torch使用(509服务器)
        torch.save(net.state_dict(), os.path.join(model_root, filename))


def printParams():
    """打印并保存算法参数"""
    print("\n")

    print("======== 【params】 ========")
    print('used_service:{}'.format(params.used_service))
    print('sample_size:{}'.format(params.sample_size))
    print('num_domains:{}'.format(params.num_domains))
    print('split_ratio(Ds:Dt=(split_ratio):(num_domains-split_ratio)):{}'.format(params.split_ratio))
    print('batch_size:{}'.format(params.batch_size))
    print('optim_name:{}'.format(params.optim_name))
    print('train_num_samples:{}'.format(params.train_num_samples))
    print('valid_num_samples:{}'.format(params.valid_num_samples))
    print("===========================")
    print('result_file_root:{}'.format(params.result_file_root))
    print('saveRoot_models:{}'.format(params.saveRoot_models))
    print("===========================")
    # print('train_dataPath_D1_T:{}'.format(params.train_dataPath_D1_T))
    # print('train_dataPath_D1_F:{}'.format(params.train_dataPath_D1_F))
    # print('train_dataPath_D2_T:{}'.format(params.train_dataPath_D2_T))
    # print('train_dataPath_D2_F:{}'.format(params.train_dataPath_D2_F))
    # print('train_dataPath_D3_T:{}'.format(params.train_dataPath_D3_T))
    # print('train_dataPath_D3_F:{}'.format(params.train_dataPath_D3_F))
    # print('train_dataPath_D4_T:{}'.format(params.train_dataPath_D4_T))
    # print('train_dataPath_D4_F:{}'.format(params.train_dataPath_D4_F))
    print('valid_dataPath_T:{}'.format(params.valid_dataPath_T))
    print('valid_dataPath_F:{}'.format(params.valid_dataPath_F))
    print("======== 【params】 ========")
    print("\n")

    # 判断目录是否存在，不存在则递归创建目录
    if not os.path.exists(params.result_file_root):
        os.makedirs(params.result_file_root)

    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    f_acc_valid = open(params.result_valid_file, "a+")
    f_acc_valid.write(
        "-----------------------------------------" + current_time + "-------------------------------------------\n")
    f_acc_valid.write(('======== 【params】 ========\n'))
    f_acc_valid.write(('used_service:' + str(params.used_service) + '\n'))
    f_acc_valid.write(('sample_size:' + str(params.sample_size) + '\n'))
    f_acc_valid.write(('num_domains:' + str(params.num_domains) + '\n'))
    f_acc_valid.write(('split_ratio(Ds:Dt=(split_ratio):(num_domains-split_ratio)):' + str(params.split_ratio) + '\n'))
    f_acc_valid.write(('batch_size:' + str(params.batch_size) + '\n'))
    f_acc_valid.write(('optim_name:' + str(params.optim_name) + '\n'))
    f_acc_valid.write(('train_num_samples:' + str(params.train_num_samples) + '\n'))
    f_acc_valid.write(('valid_num_samples:' + str(params.valid_num_samples) + '\n\n'))

    f_acc_valid.write(('result_file_root:' + str(params.result_file_root) + '\n'))
    f_acc_valid.write(('saveRoot_models:' + str(params.saveRoot_models) + '\n\n'))

    # f_acc_valid.write(('train_dataPath_D1_T:' + str(params.train_dataPath_D1_T) + '\n'))
    # f_acc_valid.write(('train_dataPath_D1_F:' + str(params.train_dataPath_D1_F) + '\n'))
    # f_acc_valid.write(('train_dataPath_D2_T:' + str(params.train_dataPath_D2_T) + '\n'))
    # f_acc_valid.write(('train_dataPath_D2_F:' + str(params.train_dataPath_D2_F) + '\n'))
    # f_acc_valid.write(('train_dataPath_D3_T:' + str(params.train_dataPath_D3_T) + '\n'))
    # f_acc_valid.write(('train_dataPath_D3_F:' + str(params.train_dataPath_D3_F) + '\n'))
    # f_acc_valid.write(('train_dataPath_D4_T:' + str(params.train_dataPath_D4_T) + '\n'))
    # f_acc_valid.write(('train_dataPath_D4_F:' + str(params.train_dataPath_D4_F) + '\n'))
    f_acc_valid.write(('valid_dataPath_T:' + str(params.valid_dataPath_T) + '\n'))
    f_acc_valid.write(('valid_dataPath_F:' + str(params.valid_dataPath_F) + '\n'))

    f_acc_valid.write(('======== 【params】 ========\n\n\n'))
    f_acc_valid.close()


def check_dataPath(dataPath):
    """
    检测图像数据的路径是否在所运行的服务器上。
    带有服务器型号标志的说明不同服务器上路径不同，需要修改；无型号标志的说明不同服务器上路径相同
    """
    if '509' in dataPath:
        out_dataPath = dataPath.replace('509', params.used_service)
    elif '3090' in dataPath:
        out_dataPath = dataPath.replace('3090', params.used_service)
    else:
        out_dataPath = dataPath
    return out_dataPath


def getResult(prob, threshold=0.5):
    """
    根据分类概率计算分类结果
    :param prob: 每个样本的分类概率，[batch_size, class_num]
    :param threshold: 0.5
    :return: 每个样本的分类结果，[batch_size]
    """
    size = prob.size()
    result = torch.zeros(size[0]).cuda()
    # result = torch.zeros(size[0], 1).cuda()
    for i in range(size[0]):
        data_0 = prob[i, 0]
        if data_0 > threshold:
            result[i] = int(0)
        else:
            result[i] = int(1)
    return result


def getResultMask(prob, threshold=0.5):
    """
    根据分类概率计算分类结果
    :param prob: 每个样本的分类概率，[batch_size, class_num=2, W, H]
    :param threshold: 0.5
    :return: 每个样本的分类结果，[batch_size, W, H]
    """
    size = prob.size()  # [batch_size, class_num=2, W, H]
    result_ori = torch.zeros(size[0], size[2], size[3]).cuda()  # [batch_size, W, H]
    result_thr = torch.zeros(size[0], size[2], size[3]).cuda()  # [batch_size, W, H]

    for i in range(size[0]):
        data_0 = prob[i, 1, :, :]  # [W, H]，类别1的通道，每个像素值表示像素伪造的概率
        tmp0 = data_0.clone()
        tmp1 = data_0.clone()

        tmp0[tmp1 >= threshold] = int(1)  # 预测为0的概率大于0.5，则判为伪造像素，值为1
        tmp0[tmp1 < threshold] = int(0)  # 预测为0的概率小于0.5，则判为真实像素，值为1

        result_ori[i, :, :] = data_0  # [W, H]
        result_thr[i, :, :] = tmp0  # [W, H]

    return result_ori, result_thr


def to_percent(y, position):
    return str(1 * y) + '%'
    # return str(y/100)


def var_to_np(img_var):
    return img_var.data.cpu().numpy()


def norm(img, min_s=0, max_s=1, min_t=0, max_t=255):
    """归一化图像，默认将[0,1]归一化到[0,255]"""
    out = ((img - min_s) / (max_s - min_s)) * (max_t - min_t) + min_t
    return out


"""模型训练相关的函数==========================================================================================="""


def cat_multiDomains(Domains_data_list):
    """将多域数据进行拼接"""
    Domain_number = len(Domains_data_list)

    # 先取出第一个域的数据
    input_data = Domains_data_list[0]
    input_face = Variable(input_data[0]).cuda()
    input_mask = Variable(input_data[1]).cuda()
    input_label = Variable(input_data[2]).cuda()
    input_Domain_ID = Variable(input_data[3]).cuda()
    input_path = input_data[4]
    input_face_next = Variable(input_data[5]).cuda()
    input_mask_next = Variable(input_data[6]).cuda()

    for i in range(Domain_number - 1):
        input_data2 = Domains_data_list[i + 1]
        input_face2 = Variable(input_data2[0]).cuda()
        input_mask2 = Variable(input_data2[1]).cuda()
        input_label2 = Variable(input_data2[2]).cuda()
        input_Domain_ID2 = Variable(input_data2[3]).cuda()
        input_path2 = input_data2[4]
        input_face_next2 = Variable(input_data2[5]).cuda()
        input_mask_next2 = Variable(input_data2[6]).cuda()

        input_face = torch.cat((input_face, input_face2), 0)
        input_mask = torch.cat((input_mask, input_mask2), 0)
        input_label = torch.cat((input_label, input_label2), 0)
        input_Domain_ID = torch.cat((input_Domain_ID, input_Domain_ID2), 0)
        input_path = input_path + input_path2  # 字符串列表合并
        input_face_next = torch.cat((input_face_next, input_face_next2), 0)
        input_mask_next = torch.cat((input_mask_next, input_mask_next2), 0)
        # print(input_face.shape)
    return input_face, input_mask, input_label, input_Domain_ID, input_path, input_face_next, input_mask_next


def imageRelighting1(input_image, input_mask, reference_image):
    """这里的input_image, input_mask, reference_image均为图像路径名，字符串类型数据"""
    """加载模型"""
    model = RelightNet()
    model.load_state_dict(torch.load(params.savePath_Relighting))
    model = model.float()
    model = model.cuda()
    model.eval()
    # print(model)

    epoch = 200
    intrinsic_matrix = np.zeros((1, 3, 3))
    intrinsic_matrix[:, 0, 0] = 700.0
    intrinsic_matrix[:, 1, 1] = 700.0
    intrinsic_matrix[:, 2, 2] = 1.0
    intrinsic_matrix[:, 0, 2] = model.img_width / 2.0
    intrinsic_matrix[:, 1, 2] = model.img_height / 2.0
    intrinsic_matrix = torch.from_numpy(intrinsic_matrix)

    with torch.no_grad():
        curr_input_image = torch.reshape(torch.from_numpy(cv2.resize(imageio.imread(input_image) / 255.0, (256, 256))),
                                         (1, 256, 256, 3))
        curr_reference_image = torch.reshape(
            torch.from_numpy(cv2.resize(imageio.imread(reference_image) / 255.0, (256, 256))), (1, 256, 256, 3))
        # curr_input_image = torch.reshape(torch.from_numpy(imageio.imread(input_image)/255.0), (1, 256, 256, 3))
        # curr_reference_image = torch.reshape(torch.from_numpy(imageio.imread(reference_image)/255.0), (1, 256, 256, 3))
        curr_training_lighting = torch.from_numpy(np.zeros((model.batch_size, 4)))
        curr_img_name = input_image
        curr_mask_fill_nose = torch.from_numpy(
            np.reshape(cv2.resize(imageio.imread(input_mask), (256, 256)), (256, 256, 1))) / 255.0
        # curr_mask_fill_nose = torch.from_numpy(np.reshape(imageio.imread(input_mask), (256, 256, 1)))/255.0
        albedo, depth, shadow_mask_weights, ambient_light, full_shading, rendered_images, unit_light_direction, ambient_values, final_shading, surface_normals, estimated_unit_light_direction, estimated_ambient_light = model(
            curr_reference_image.float().cuda(), epoch, intrinsic_matrix.cuda(), curr_mask_fill_nose.cuda(),
            torch.reshape(curr_training_lighting[:, 1:4].float().cuda(), (model.batch_size, 3, 1, 1)),
            torch.reshape(curr_training_lighting[:, 0].float().cuda(), (model.batch_size, 1, 1)))

        albedo, depth, shadow_mask_weights, ambient_light, full_shading, rendered_images, unit_light_direction, ambient_values, final_shading, surface_normals, estimated_unit_light_direction, estimated_ambient_light = model(
            curr_input_image.float().cuda(), epoch, intrinsic_matrix.cuda(), curr_mask_fill_nose.cuda(),
            torch.reshape(estimated_unit_light_direction.float().cuda(), (model.batch_size, 3, 1, 1)),
            torch.reshape(estimated_ambient_light.float().cuda(), (model.batch_size, 1, 1)))

        rendered_images = rendered_images.permute(0, 2, 3, 1)
        rendered_images = rendered_images.cpu().numpy()
        albedo = albedo.permute(0, 2, 3, 1)
        albedo = albedo.cpu().numpy()
        depth = depth.permute(0, 2, 3, 1)
        depth = depth.cpu().numpy()
        depth = -depth
        depth = (depth - np.amin(depth)) / (np.amax(depth) - np.amin(depth))

        final_shading = final_shading.cpu().numpy()

        surface_normals = surface_normals.permute(0, 2, 3, 1)
        surface_normals = surface_normals.cpu().numpy()
        surface_normals = 255.0 * (surface_normals + 1.0) / 2.0

        curr_mask_fill_nose_3_channels = np.zeros((model.img_height, model.img_width, 3))
        curr_mask_fill_nose_3_channels[:, :, 0] = np.reshape(curr_mask_fill_nose.numpy(),
                                                             (model.img_height, model.img_width))
        curr_mask_fill_nose_3_channels[:, :, 1] = np.reshape(curr_mask_fill_nose.numpy(),
                                                             (model.img_height, model.img_width))
        curr_mask_fill_nose_3_channels[:, :, 2] = np.reshape(curr_mask_fill_nose.numpy(),
                                                             (model.img_height, model.img_width))

        name_parts = curr_img_name.split('.')  # ['Deepfake_test_images/00295_img', 'png']
        print(name_parts[0])
        name_parts = name_parts[0].split('/')  # ['Deepfake_test_images', '00295_img']
        print(name_parts[0])

        input_image = curr_input_image[0].detach().cpu().numpy() * 255.0
        input_image = input_image[:, :, ::-1]
        rendered_image = 255.0 * rendered_images[0, :, :, ::-1] * curr_mask_fill_nose_3_channels

        input_image[curr_mask_fill_nose_3_channels > 0] = rendered_image[curr_mask_fill_nose_3_channels > 0]
        print(input_image.shape)  # [256,256,3]

        """保存输出的相关图像"""
        if 0:
            cv2.imwrite('lighting_transfer_result/' + name_parts[0] + '_rendered_image.png', input_image)
            cv2.imwrite('lighting_transfer_result/' + name_parts[0] + '_shadow_mask.png',
                        255.0 * shadow_mask_weights[0, :, :].cpu().numpy() * np.reshape(curr_mask_fill_nose.numpy(), (
                            model.img_height, model.img_width)))
            cv2.imwrite('lighting_transfer_result/' + name_parts[0] + '_albedo.png',
                        255.0 * albedo[0, :, :, ::-1] * curr_mask_fill_nose_3_channels)
            cv2.imwrite('lighting_transfer_result/' + name_parts[0] + '_depth.png',
                        255.0 * depth[0, :, :, :] * curr_mask_fill_nose.numpy())
            cv2.imwrite('lighting_transfer_result/' + name_parts[0] + '_shading.png',
                        255.0 * final_shading[0, :, :] * np.reshape(curr_mask_fill_nose.numpy(),
                                                                    (model.img_height, model.img_width)))
            cv2.imwrite('lighting_transfer_result/' + name_parts[0] + '_surface_normals.png',
                        surface_normals[0, :, :, ::-1] * curr_mask_fill_nose_3_channels)
            print('\nSaving Finished!\n')
        return input_image


def imageRelighting2(input_image, input_mask, reference_image, batch_size=1, devices="cuda:0"):
    """这里的input_image, input_mask, reference_image均为tensor图像数据，数据维度为[b,c,h,w]"""
    """加载模型"""
    # b,c,w,h = input_image.shape
    # print(b)
    # model = RelightNet(batch_size=b)

    model = RelightNet(batch_size=batch_size, device=devices)
    model.load_state_dict(torch.load(params.savePath_Relighting))
    model = model.float()
    model = model.to(devices)
    model.eval()
    # print(model)

    epoch = 200
    intrinsic_matrix = np.zeros((1, 3, 3))
    # intrinsic_matrix = np.zeros((b, 3, 3))
    intrinsic_matrix[:, 0, 0] = 700.0
    intrinsic_matrix[:, 1, 1] = 700.0
    intrinsic_matrix[:, 2, 2] = 1.0
    intrinsic_matrix[:, 0, 2] = model.img_width / 2.0
    intrinsic_matrix[:, 1, 2] = model.img_height / 2.0
    intrinsic_matrix = torch.from_numpy(intrinsic_matrix)

    with torch.no_grad():
        curr_input_image = (input_image.permute(0, 2, 3, 1) / 255.0).to(devices)
        curr_reference_image = (reference_image.permute(0, 2, 3, 1) / 255.0).to(devices)
        curr_mask_fill_nose = (input_mask.permute(0, 2, 3, 1) / 255.0).to(devices)
        curr_training_lighting = torch.from_numpy(np.zeros((model.batch_size, 4))).to(devices)

        albedo, depth, shadow_mask_weights, ambient_light, full_shading, rendered_images, unit_light_direction, ambient_values, final_shading, surface_normals, estimated_unit_light_direction, estimated_ambient_light = model(
            curr_reference_image.float().to(devices), epoch, intrinsic_matrix.to(devices),
            curr_mask_fill_nose.to(devices),
            torch.reshape(curr_training_lighting[:, 1:4].float().to(devices), (model.batch_size, 3, 1, 1)),
            torch.reshape(curr_training_lighting[:, 0].float().to(devices), (model.batch_size, 1, 1)))

        albedo, depth, shadow_mask_weights, ambient_light, full_shading, rendered_images, unit_light_direction, ambient_values, final_shading, surface_normals, estimated_unit_light_direction, estimated_ambient_light = model(
            curr_input_image.float().to(devices), epoch, intrinsic_matrix.to(devices), curr_mask_fill_nose.to(devices),
            torch.reshape(estimated_unit_light_direction.float().to(devices), (model.batch_size, 3, 1, 1)),
            torch.reshape(estimated_ambient_light.float().to(devices), (model.batch_size, 1, 1)))
        rendered_images = rendered_images.permute(0, 2, 3, 1)  # [b,c,h,w]->[b,h,w,c]

        curr_mask_fill_nose_3_channels = torch.zeros(rendered_images.shape).to(devices)
        curr_mask_fill_nose_3_channels[:, :, :, 0] = curr_mask_fill_nose.squeeze()
        curr_mask_fill_nose_3_channels[:, :, :, 1] = curr_mask_fill_nose.squeeze()
        curr_mask_fill_nose_3_channels[:, :, :, 2] = curr_mask_fill_nose.squeeze()
        input_image = curr_input_image.detach() * 255.0
        # input_image = input_image[:, :, :, [2, 1, 0]]
        rendered_image = 255.0 * rendered_images * curr_mask_fill_nose_3_channels
        input_image[curr_mask_fill_nose_3_channels > 0] = rendered_image[curr_mask_fill_nose_3_channels > 0]
        input_image = input_image.permute(0, 3, 1, 2)
        """保存输出的相关图像，1保存，0不保存"""
        albedo = albedo.permute(0, 2, 3, 1)
        depth = depth.permute(0, 2, 3, 1)
        depth = -depth
        depth = (depth - torch.amin(depth)) / (torch.amax(depth) - torch.amin(depth))

        surface_normals = surface_normals.permute(0, 2, 3, 1)
        surface_normals = 255.0 * (surface_normals + 1.0) / 2.0
        shadow_mask = 255.0 * shadow_mask_weights.to(devices) * curr_mask_fill_nose.squeeze().to(devices)
        shadow_mask = shadow_mask.unsqueeze(1)
        albedo = 255.0 * albedo * curr_mask_fill_nose_3_channels
        albedo = albedo.permute(0, 3, 1, 2)
        depth = 255.0 * depth * curr_mask_fill_nose
        depth = depth.permute(0, 3, 1, 2)
        shading = 255.0 * final_shading * curr_mask_fill_nose.squeeze()
        shading = shading.unsqueeze(1)
        surface_normals = surface_normals * curr_mask_fill_nose_3_channels
        surface_normals = surface_normals.permute(0, 3, 1, 2)

        return input_image, shadow_mask, albedo, depth, shading, surface_normals

def imageRelighting_plot(input_image, input_mask, reference_image, batch_size=1, devices="cuda:0"):
    # b,c,w,h = input_image.shape
    # print(b)
    # model = RelightNet(batch_size=b)

    model = RelightNet(batch_size=batch_size, device=devices)
    model.load_state_dict(torch.load(params.savePath_Relighting))
    model = model.float()
    model = model.to(devices)
    model.eval()
    # print(model)

    epoch = 200
    intrinsic_matrix = np.zeros((1, 3, 3))
    # intrinsic_matrix = np.zeros((b, 3, 3))
    intrinsic_matrix[:, 0, 0] = 700.0
    intrinsic_matrix[:, 1, 1] = 700.0
    intrinsic_matrix[:, 2, 2] = 1.0
    intrinsic_matrix[:, 0, 2] = model.img_width / 2.0
    intrinsic_matrix[:, 1, 2] = model.img_height / 2.0
    intrinsic_matrix = torch.from_numpy(intrinsic_matrix)
    out = []
    with torch.no_grad():
        lighting_configs = [
            # 顶部光 (原文代码已有的示例)
            {'name': 'A00E45',    'vec': [0.0, 0.7071, 0.7071]}, 
            # 左侧光 (Y取负值)
            {'name': 'A60E-20', 'vec': [-0.8138, -0.3420, 0.4698]},
            # 右侧光 (X取负值)
            {'name': 'A-60E-20',   'vec': [0.8138, -0.3420, 0.4698]},
            # 右侧光 (X取正值)
            {'name': 'Multi-PIE_18',  'vec': [-0.7076, 0.3892, 0.5897]}
        ]
        training_lightings = np.zeros((batch_size, 4))
        training_lightings[:, 0] = 0.5
        for config in lighting_configs:
            training_lightings[:, 1] = config['vec'][0]
            training_lightings[:, 2] = config['vec'][1]
            training_lightings[:, 3] = config['vec'][2]
            curr_input_image = (input_image.permute(0, 2, 3, 1) / 255.0).to(devices)
            curr_reference_image = (reference_image.permute(0, 2, 3, 1) / 255.0).to(devices)
            curr_mask_fill_nose = (input_mask.permute(0, 2, 3, 1) / 255.0).to(devices)
            # curr_training_lighting = torch.from_numpy(np.zeros((model.batch_size, 4))).to(devices)
            curr_training_lighting = torch.from_numpy(training_lightings).to(devices)

            # albedo, depth, shadow_mask_weights, ambient_light, full_shading, rendered_images, unit_light_direction, ambient_values, final_shading, surface_normals, estimated_unit_light_direction, estimated_ambient_light = model(
            #     curr_reference_image.float().to(devices), epoch, intrinsic_matrix.to(devices),
            #     curr_mask_fill_nose.to(devices),
            #     torch.reshape(curr_training_lighting[:, 1:4].float().to(devices), (model.batch_size, 3, 1, 1)),
            #     torch.reshape(curr_training_lighting[:, 0].float().to(devices), (model.batch_size, 1, 1)))
            
            albedo, depth, shadow_mask_weights, ambient_light, full_shading, rendered_images, unit_light_direction, ambient_values, final_shading, surface_normals, estimated_unit_light_direction, estimated_ambient_light = model(
                curr_input_image.float().to(devices), epoch, intrinsic_matrix.to(devices), curr_mask_fill_nose.to(devices),
                torch.reshape(curr_training_lighting[:, 1:4].float().to(devices), (model.batch_size, 3, 1, 1)),
                torch.reshape(curr_training_lighting[:, 0].float().to(devices), (model.batch_size, 1, 1)))
            
            # albedo, depth, shadow_mask_weights, ambient_light, full_shading, rendered_images, unit_light_direction, ambient_values, final_shading, surface_normals, estimated_unit_light_direction, estimated_ambient_light = model(
            #     curr_input_image.float().to(devices), epoch, intrinsic_matrix.to(devices), curr_mask_fill_nose.to(devices),
            #     torch.reshape(estimated_unit_light_direction.float().to(devices), (model.batch_size, 3, 1, 1)),
            #     torch.reshape(estimated_ambient_light.float().to(devices), (model.batch_size, 1, 1)))
            rendered_images = rendered_images.permute(0, 2, 3, 1)  # [b,c,h,w]->[b,h,w,c]

            curr_mask_fill_nose_3_channels = torch.zeros(rendered_images.shape).to(devices)
            curr_mask_fill_nose_3_channels[:, :, :, 0] = curr_mask_fill_nose.squeeze()
            curr_mask_fill_nose_3_channels[:, :, :, 1] = curr_mask_fill_nose.squeeze()
            curr_mask_fill_nose_3_channels[:, :, :, 2] = curr_mask_fill_nose.squeeze()
            input_image = curr_input_image.detach() * 255.0
            # input_image = input_image[:, :, :, [2, 1, 0]]
            rendered_image = 255.0 * rendered_images * curr_mask_fill_nose_3_channels
            input_image[curr_mask_fill_nose_3_channels > 0] = rendered_image[curr_mask_fill_nose_3_channels > 0]
            input_image = input_image.permute(0, 3, 1, 2)
            albedo = albedo.permute(0, 2, 3, 1)
            depth = depth.permute(0, 2, 3, 1)
            depth = -depth
            depth = (depth - torch.amin(depth)) / (torch.amax(depth) - torch.amin(depth))

            surface_normals = surface_normals.permute(0, 2, 3, 1)
            surface_normals = 255.0 * (surface_normals + 1.0) / 2.0
            shadow_mask = 255.0 * shadow_mask_weights.to(devices) * curr_mask_fill_nose.squeeze().to(devices)
            shadow_mask = shadow_mask.unsqueeze(1)
            albedo = 255.0 * albedo * curr_mask_fill_nose_3_channels
            albedo = albedo.permute(0, 3, 1, 2)
            depth = 255.0 * depth * curr_mask_fill_nose
            depth = depth.permute(0, 3, 1, 2)
            shading = 255.0 * final_shading * curr_mask_fill_nose.squeeze()
            shading = shading.unsqueeze(1)
            surface_normals = surface_normals * curr_mask_fill_nose_3_channels
            surface_normals = surface_normals.permute(0, 3, 1, 2)
            out.append({"data":[input_image, shadow_mask, albedo, depth, shading, surface_normals], "direction": config['name']})
        return out

def relight_plot(train_dataloader_list):
    batch_num = int(params.train_num_samples / params.sample_size)
    print('\nbatch_num:{}\n'.format(batch_num))
    for epoch in range(params.num_epochs):
        data_zip = zip(*train_dataloader_list)

        progress_bar = tqdm(enumerate(data_zip), total=batch_num,
                            desc=f"Epoch [{epoch + 1}/{params.num_epochs}]",
                            unit="batch")

        for batch_index, data_list_ori in progress_bar:
            """
            将多域数据作为原始数据，打乱后的数据用于对比学习模块
            """
            face_ori, mask_ori, label_ori, Domain_ID_ori, path_ori, input_face_next, input_mask_next = cat_multiDomains(data_list_ori)
            mask_ori_re = torch.where(mask_ori == 0, 1, 0)
            mask_ones = torch.ones(mask_ori.shape).to(mask_ori.device) * 255
            out = imageRelighting_plot(
                input_image=face_ori,
                input_mask=mask_ones,
                reference_image=face_ori,
                batch_size=face_ori.shape[0],
                devices="cuda:0")
            data_name = [
                "df_true", "df", "fs_true", "fs", "f2f_true", "f2f", "nt_true", "nt",
                "df_true_sbi", "df_false_sbi", "fs_true_sbi", "fs_false_sbi", "f2f_true_sbi", "f2f_false_sbi", "nt_true_sbi", "nt_false_sbi"
                ]
            for data in out:
                face_tar, shadow_mask, albedo, depth, shading, surface_normals = data["data"]
                direction = data["direction"]
                for ide, i in enumerate(range(face_ori.shape[0])):
                    if ide < 2:
                        name = data_name[0]
                    elif ide < 4:
                        name = data_name[1]
                    elif ide < 6:
                        name = data_name[2]
                    elif ide < 8:
                        name = data_name[3]
                    elif ide < 10:
                        name = data_name[4]
                    elif ide < 12:
                        name = data_name[5]
                    elif ide < 14:
                        name = data_name[6]
                    elif ide < 16:
                        name = data_name[7]
                    elif ide < 18:
                        name = data_name[8]
                    elif ide < 20:
                        name = data_name[9]
                    elif ide < 22:
                        name = data_name[10]
                    elif ide < 24:
                        name = data_name[11]
                    elif ide < 26:
                        name = data_name[12]
                    elif ide < 28:
                        name = data_name[13]
                    elif ide < 30:
                        name = data_name[14]
                    else:
                        name = data_name[15]
                    
                    ori = face_ori.permute(0, 2, 3, 1).cpu().numpy()[i, :, :, :].astype(np.uint8)
                    ma_ori = mask_ori.permute(0, 2, 3, 1).cpu().numpy()[i, :, :, :].astype(np.uint8)
                    tar = face_tar.permute(0, 2, 3, 1).cpu().numpy()[i, :, :, :].astype(np.uint8)
                    sh_mask = shadow_mask.permute(0, 2, 3, 1).cpu().numpy()[i, :, :, :].astype(np.uint8)
                    al = albedo.permute(0, 2, 3, 1).cpu().numpy()[i, :, :, :].astype(np.uint8)
                    de = depth.permute(0, 2, 3, 1).cpu().numpy()[i, :, :, :].astype(np.uint8)
                    shad = shading.permute(0, 2, 3, 1).cpu().numpy()[i, :, :, :].astype(np.uint8)
                    surface = surface_normals.permute(0, 2, 3, 1).cpu().numpy()[i, :, :, :].astype(np.uint8)
                    ma_ori_re = mask_ori_re.permute(0, 2, 3, 1).cpu().numpy()[i, :, :, :].astype(np.uint8)
                    tar_mask = tar * ma_ori
                    ori_reshape = tar_mask + ori * ma_ori_re
                    images = [ori, ma_ori, tar, sh_mask, al, de, shad, surface, ori_reshape]
                    titles = ["Original Face", "Original Mask", "Target Face", "Shadow Mask", "Albedo",
                              "Depth", "Shading", "Surface Normals", "Original Reshape"]
                    fig = plt.figure(figsize=(16, 10))
                    grid = plt.GridSpec(2, 5, wspace=0.3, hspace=0.3)
                    positions = [
                        grid[0, 0], grid[0, 1], grid[0, 2], grid[0, 3], grid[0, 4],
                        grid[1, 0], grid[1, 1], grid[1, 2], grid[1, 3]
                    ]
                    for idx, (img, title, pos) in enumerate(zip(images, titles, positions)):
                        ax = fig.add_subplot(pos)
                        if img.shape[2] == 3:
                            ax.imshow(img)
                        elif img.shape[2] == 1:
                            ax.imshow(img[:, :, 0], cmap="gray")
                        ax.set_title(title, fontsize=12, fontweight="bold")
                        ax.axis("off")
                    ax = fig.add_subplot(grid[1, 4])
                    ax.axis("off")
                    number = len(os.listdir("./data/relight/" + direction + "/" + name))
                    plt.savefig(os.path.join('./data/relight/' + direction, name, str(i) + "_" + str(number) + '_all.jpg'))
                    ma_ori = np.squeeze(ma_ori)
                    sh_mask = np.squeeze(ma_ori)
                    shad = np.squeeze(shad)
                    de = np.squeeze(de)
                    Image.fromarray(ori).save(os.path.join("./data/relight/" + direction, name, str(i) + "_" + str(number) + "_ori.jpg"))
                    Image.fromarray(ma_ori * 255).save(os.path.join("./data/relight/" + direction, name, str(i) + "_" + str(number) + "_ori_mask.jpg"))
                    Image.fromarray(tar).save(os.path.join("./data/relight/" + direction, name, str(i) + "_" + str(number) + "_tar.jpg"))
                    Image.fromarray(sh_mask).save(os.path.join("./data/relight/" + direction, name, str(i) + "_" + str(number) + "_shadow_mask.jpg"))
                    Image.fromarray(al).save(os.path.join("./data/relight/" + direction, name, str(i) + "_" + str(number) + "_albedo.jpg"))
                    Image.fromarray(de).save(os.path.join("./data/relight/" + direction, name, str(i) + "_" + str(number) + "_depth.jpg"))
                    Image.fromarray(shad).save(os.path.join("./data/relight/" + direction, name, str(i) + "_" + str(number) + "_shading.jpg"))
                    Image.fromarray(surface).save(os.path.join("./data/relight/" + direction, name, str(i) + "_" + str(number) + "_surface_normals.jpg"))
                    Image.fromarray(ori_reshape).save(os.path.join("./data/relight/" + direction, name, str(i) + "_" + str(number) + "_ori_reshape.jpg"))
                    print(f"out {os.path.join('./data/relight/' + direction, name, str(i) + '_' + str(number) + '_all.jpg')}")

def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()


class AMSoftmaxLoss(nn.Module):
    """Computes the AM-Softmax loss with cos or arc margin"""
    margin_types = ['cos', 'arc']

    def __init__(self, margin_type='cos', gamma=0., m=0.5, s=30, t=1.):
        super(AMSoftmaxLoss, self).__init__()
        assert margin_type in AMSoftmaxLoss.margin_types
        self.margin_type = margin_type
        assert gamma >= 0
        self.gamma = gamma
        assert m > 0
        self.m = m
        assert s > 0
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        assert t >= 1
        self.t = t

    def forward(self, cos_theta, target):
        if self.margin_type == 'cos':
            phi_theta = cos_theta - self.m
        else:
            sine = torch.sqrt(1.0 - torch.pow(cos_theta, 2))
            phi_theta = cos_theta * self.cos_m - sine * self.sin_m #cos(theta+m)
            phi_theta = torch.where(cos_theta > self.th, phi_theta, cos_theta - self.sin_m * self.m)

        index = torch.zeros_like(cos_theta, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        output = torch.where(index, phi_theta, cos_theta)

        if self.gamma == 0 and self.t == 1.:
            return F.cross_entropy(self.s*output, target)

        if self.t > 1:
            h_theta = self.t - 1 + self.t*cos_theta
            support_vecs_mask = (1 - index) * \
                torch.lt(torch.masked_select(phi_theta, index).view(-1, 1).repeat(1, h_theta.shape[1]) - cos_theta, 0)
            output = torch.where(support_vecs_mask, h_theta, output)
            return F.cross_entropy(self.s*output, target)

        return focal_loss(F.cross_entropy(self.s*output, target, reduction='none'), self.gamma)

def freeze_aug_components(model):
    # [核心修改] 检查是否包裹了 DataParallel 或 DistributedDataParallel
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        real_model = model.module
    else:
        real_model = model
        
    print("正在冻结 features_aug 分支涉及的组件...")
    
    # 注意：下面所有的 model. 都要改成 real_model.
    
    # 1. 冻结共享的骨干网络
    modules_to_freeze = [
        real_model.xception_rgb, 
        real_model.xception_srm, 
        real_model.xception_PLGF,
        real_model.PLGF_descriptor,
    ]

    # 2. 冻结共享的预处理/卷积层
    modules_to_freeze.extend([
        real_model.srm_conv0, real_model.srm_conv1, real_model.srm_conv2,
        real_model.PLGF_conv1, real_model.PLGF_conv2,
    ])

    # 3. 冻结共享的注意力与融合模块
    modules_to_freeze.extend([
        real_model.srm_sa, real_model.PLGF_sa, real_model.srm_sa_post,
        real_model.dual_cma0, real_model.dual_cma1,
        real_model.fusion,
    ])

    # 4. 冻结 features_aug 独有的融合层
    modules_to_freeze.extend([
        real_model.att13, real_model.att14, real_model.att15,
        real_model.att16, real_model.att17, real_model.att18
    ])

    # 5. 冻结最后的分类头
    modules_to_freeze.extend([
        real_model.end_linear, 
        real_model.seg_decoder
    ])

    # --- 执行冻结操作 ---
    for module in modules_to_freeze:
        for param in module.parameters():
            param.requires_grad = False
            
    print("冻结完成。")

def train_model(model, train_dataloader_list, valid_dataloader_list):
    """定义优化器"""
    optim_name = params.optim_name
    # freeze_aug_components(model=model)
    if optim_name == 'SGD':
        # 学习率固定的方法，最好结合学习率衰减的策略
        optimizer_model = torch.optim.SGD(model.parameters(), lr=params.model_SGD_lr, momentum=0.9, weight_decay=0.0005)
        # 学习率衰减策略：衰减周期为30epoch，初始lr=0.01，每个epoch后学习率lr=lr*0.2
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer_model, step_size=1, gamma=0.8)
    else:
        # 自适应学习率方法，可以不用添加学习率衰减策略，lr=0.0001
        optimizer_model = optim.Adam(model.parameters(), lr=params.model_Adam_lr, betas=(0.9, 0.999),
                                     weight_decay=0.0005)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer_model, step_size=1, gamma=0.7)

    """损失函数"""
    loss_cls = nn.CrossEntropyLoss()  # 真假分类loss，用于判别真假标签
    loss_cls2 = nn.CrossEntropyLoss()  # 真假分类loss，用于判别relight后真假标签
    loss_cls3 = nn.CrossEntropyLoss()  # 真假分类loss，用于判别relight后真假标签
    loss_contra = ContrastLoss()  # 对比学习loss，用于解耦风格特征
    loss_contra2 = ContrastLoss2()
    loss_adv = nn.CrossEntropyLoss()  # 对抗学习loss，用于判别域标签
    loss_am = AMSoftmaxLoss()
    loss_am2 = AMSoftmaxLoss()
    alpha1, alpha2, alpha3 = 1, 1, 1

    # 定义loss统计工具，用于返回loss的均值和标准差
    loss_meter_cls = meter.AverageValueMeter()
    loss_meter_cls2 = meter.AverageValueMeter()
    loss_meter_contra = meter.AverageValueMeter()
    loss_meter_adv = meter.AverageValueMeter()
    loss_meter_all = meter.AverageValueMeter()

    softmax = torch.nn.Softmax(dim=1)  # 计算分类概率，dim=1表示按行计算

    # 判断result.txt的目录是否存在，不存在则递归创建目录
    if not os.path.exists(params.result_file_root):
        os.makedirs(params.result_file_root)
    # 判断result_figures_path的目录是否存在，不存在则递归创建目录
    if not os.path.exists(params.result_figures_path):
        os.makedirs(params.result_figures_path)

    batch_num = int(params.train_num_samples / params.sample_size)
    print('\nbatch_num:{}\n'.format(batch_num))
    """训练模型"""
    for epoch in range(params.num_epochs):
        # 模型调整为训练模式
        model.train()

        # 每个epoch都重置统计信息工具
        loss_meter_cls.reset()
        loss_meter_cls2.reset()
        loss_meter_contra.reset()
        loss_meter_adv.reset()
        loss_meter_all.reset()
        confusion_matrix = torch.zeros(2, 2)

        data_zip = zip(*train_dataloader_list)

        current_lr = optimizer_model.param_groups[0]['lr']

        progress_bar = tqdm(enumerate(data_zip), total=batch_num,
                            desc=f"Epoch [{epoch + 1}/{params.num_epochs}] LR: {current_lr:.6f}",
                            unit="batch")

        for batch_index, data_list_ori in progress_bar:
            """
            将多域数据作为原始数据，打乱后的数据用于对比学习模块
            """
            face_ori, mask_ori, label_ori, Domain_ID_ori, path_ori, input_face_next, input_mask_next = cat_multiDomains(data_list_ori)

            # 随机打乱得到用于对比学习的数据
            number = len(data_list_ori)
            batch_size = face_ori.shape[0]
            not_aug_batch = int(batch_size/number * 8)

            face_ori_not_aug = face_ori[0:not_aug_batch, :, :, :]
            face_ori_aug = face_ori[not_aug_batch:batch_size, :, :, :]

            mask_ori_not_aug = mask_ori[0:not_aug_batch, :, :, :]
            mask_ori_aug = mask_ori[not_aug_batch:batch_size, :, :, :]

            label_ori_not_aug = label_ori[0:not_aug_batch]
            label_ori_aug = label_ori[not_aug_batch:batch_size]

            face_ori_next_not_aug = input_face_next[0:not_aug_batch, :, :, :]
            mask_ori_next_not_aug = input_mask_next[not_aug_batch:batch_size, :, :, :]

            rand_idx_not_aug = torch.randperm(face_ori_not_aug.shape[0])
            rand_idx_aug = torch.randperm(face_ori_aug.shape[0])

            face_ori_not_aug = face_ori_not_aug[rand_idx_not_aug]
            face_ori_aug = face_ori_aug[rand_idx_aug]

            mask_ori_not_aug = mask_ori_not_aug[rand_idx_not_aug]
            mask_ori_aug = mask_ori_aug[rand_idx_aug]

            label_ori_not_aug = label_ori_not_aug[rand_idx_not_aug]
            label_ori_aug = label_ori_aug[rand_idx_aug]

            face_ori_next_not_aug = face_ori_next_not_aug[rand_idx_not_aug]
            mask_ori_next_not_aug = mask_ori_next_not_aug[rand_idx_aug]

            mask_ori_not_aug = (mask_ori_not_aug > 0.5).long().squeeze(1)
            mask_ori_aug = (mask_ori_aug > 0.5).long().squeeze(1)


            """训练模型"""
            # 梯度清零，否则梯度是累加的
            optimizer_model.zero_grad()
            face_ori_not_aug = face_ori_not_aug.cuda()
            face_ori_aug = face_ori_aug.detach().cuda()
            face_ori_next_not_aug = face_ori_next_not_aug.cuda()

            
            out = model(face_ori_not_aug, face_ori_aug, face_ori_next_not_aug)
            am_out1 = out[0]
            mask_out1 = out[1]
            am_out2 = out[2]
            mask_out2 = out[3]


            label_ori_not_aug = label_ori_not_aug.to(am_out1.device)
            am_loss = loss_am(am_out1, label_ori_not_aug)  # 分类loss，只用到Ds的数据，因此使用Ds的label

            label_ori_aug = label_ori_aug.to(am_out1.device)
            am_loss2 = loss_am2(am_out2, label_ori_aug)  # 分类loss，只用到Ds的数据，因此使用Ds的label

            mask_ori_not_aug = mask_ori_not_aug.to(mask_out1.device)
            mask_loss1 = loss_cls(mask_out1, mask_ori_not_aug)

            mask_ori_aug = mask_ori_aug.to(mask_out1.device)
            mask_loss2 = loss_cls2(mask_out2, mask_ori_aug)

            
            pred_cls = torch.cat([am_out1, am_out2], dim=0)
            label_ori = torch.cat([label_ori_not_aug, label_ori_aug], dim=0)

            loss = am_loss + 0.1 * am_loss2 + 0.01 * mask_loss1 + 0.001 * mask_loss2
            cls_loss = am_loss
            cls_loss2 = am_loss2
            contra_loss = mask_loss1
            adv_loss = mask_loss2
            loss.backward()
            optimizer_model.step()  # 更新分类网络参数

            """更新统计指标"""
            loss_meter_all.add(loss.item())
            loss_meter_cls.add(cls_loss.item())
            loss_meter_cls2.add(cls_loss2.item())
            loss_meter_contra.add(contra_loss.item())
            loss_meter_adv.add(adv_loss.item())
            loss_all_mean = loss_meter_all.value()[0]
            loss_cls_mean = loss_meter_cls.value()[0]
            loss_cls2_mean = loss_meter_cls2.value()[0]
            loss_contra_mean = loss_meter_contra.value()[0]
            loss_adv_mean = loss_meter_adv.value()[0]

            """计算分类结果"""
            preds = softmax(pred_cls)
            result_index = getResult(preds, threshold=0.5)  # 根据分类概率计算分类结果
            result_index = result_index.long()
            train_label = label_ori.long()
            result_index = result_index
            train_label = train_label.to(result_index.device)
            for j in range(len(train_label)):
                if train_label[j] == result_index[j]:
                    confusion_matrix[train_label[j]][train_label[j]] += 1
                else:
                    confusion_matrix[train_label[j]][result_index[j]] += 1
            # 混淆矩阵位置调整
            TP = confusion_matrix[1][1]
            TN = confusion_matrix[0][0]
            FN = confusion_matrix[1][0]
            FP = confusion_matrix[0][1]
            acc_train = float(100.0 * ((TP + TN) / (TP + TN + FN + FP)))
            TPR = 100.0 * (TP / (TP + FN))
            FPR = 100.0 * (FP / (FP + TN))
            TNR = 100 - FPR
            HTER = 100 - (TPR + TNR) / 2
            progress_bar.set_postfix({
                "Loss_all": f"{loss:.4f}",
                "Cls": f"{cls_loss:.4f}",
                "Cls2": f"{cls_loss2:.4f}",
                "Contra": f"{contra_loss:.4f}",
                "Adv": f"{adv_loss:.4f}",
                "Acc(%)": f"{acc_train:.2f}",
                "LR": f"{current_lr}"
            })

            """输出训练过程的相关信息"""
            if ((batch_index + 1) % params.print_freq == 0):
                print(
                    "Epoch [{}/{}], batch_index [{}/{}], loss_all_mean={:.6f}, loss_cls_mean={:.6f}, loss_cls2_mean={:.6f}, loss_contra_mean={:.6f}, loss_adv_mean={:.6f}, acc={:.4f}".format(
                        epoch + 1, params.num_epochs, batch_index + 1, batch_num, loss_all_mean, loss_cls_mean,
                        loss_cls2_mean,
                        loss_contra_mean, loss_adv_mean, acc_train))
                print('【confusion_matrix】:')
                print(torch.tensor([[TP, FN], [FP, TN]]))
                print('\n')

        """保存模型"""
        if (epoch + 1) % 1 == 0:
            save_model(net=model, root=params.saveRoot_Illumination,
                       filename="Illumination_epoch{}.pth".format(epoch + 1))
        torch.cuda.empty_cache()
        """模型验证"""
        f_acc_valid = open(params.result_valid_file, "a+")
        if (epoch + 1) % 1 == 0:
            print("====== valid(eval-mode) =====")
            model.eval()
            confusion_matrix = eval_model(model, valid_dataloader_list)
            name_list = ["DFD   ", 
                         "CDF-V2", 
                         "DFDC  ", 
                         "Deeper",
                         "DFDC-P"]
            for i, name in enumerate(name_list):
                TP = confusion_matrix[i][1][1]
                TN = confusion_matrix[i][0][0]
                FN = confusion_matrix[i][1][0]
                FP = confusion_matrix[i][0][1]
                acc_test = float(100.0 * ((TP + TN) / (TP + TN + FN + FP)))
                TPR = 100.0 * (TP / (TP + FN))
                FPR = 100.0 * (FP / (FP + TN))
                TNR = 100 - FPR
                HTER = 100 - (TPR + TNR) / 2
                print('【confusion_matrix】:')
                print(torch.tensor([[TP, FN], [FP, TN]]))
                print("valid({}) : Epoch [{}/{}], ACC={}, TPR={}, FPR={}, TNR={}, HTER={}".format(
                    name, epoch + 1, params.num_epochs, float(acc_test), float(TPR), float(FPR), float(TNR), float(HTER)))
                f_acc_valid.write(('epoch' + str(epoch + 1) + ', ' + name +", "+ "ACC:" + str(float(acc_test)) + ', ' + "TPR:" + str(
                    float(TPR)) + ', ' + "FPR:" + str(float(FPR)) + ', ' + "TNR:" + str(float(TNR)) + ', ' + "HTER:" + str(float(HTER)) + '\n'))
        f_acc_valid.close()

        """学习率衰减"""
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]  # 获取更新后的学习率
            print(f'epoch {epoch + 1}, {optim_name} lr: {current_lr:.6f}')

class LinearDecayLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, n_epoch, start_decay, last_epoch=-1):
        self.start_decay=start_decay
        self.n_epoch=n_epoch
        super(LinearDecayLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        last_epoch = self.last_epoch
        n_epoch=self.n_epoch
        b_lr=self.base_lrs[0]
        start_decay=self.start_decay
        if last_epoch>start_decay:
            lr=b_lr-b_lr/(n_epoch-start_decay)*(last_epoch-start_decay)
        else:
            lr=b_lr
        return [lr]


def eval_model(model, valid_dataloader_list):
    """模型验证"""

    # 定义统计指标
    confusion_matrix = torch.zeros(len(valid_dataloader_list), 2, 2)
    softmax = torch.nn.Softmax(dim=1)  # 计算分类概率，dim=1表示按行计算
    for i, data in enumerate(valid_dataloader_list):
        data_zip = zip(data[0], data[1])
        with torch.no_grad():
            for batch_index, (D1_T_data, D1_F_data) in enumerate(tqdm(data_zip)):

                data_list_ori = [D1_T_data, D1_F_data]
                face_ori, mask_ori, label_ori, Domain_ID_ori, path_ori, input_face_next, input_mask_next = cat_multiDomains(data_list_ori)

                face_tar = face_ori.detach().clone()

                out = model(face_ori, face_tar, input_face_next)
                pred_cls = out[0]

                """计算分类结果"""
                preds = softmax(pred_cls)
                result_index = getResult(preds, threshold=0.5)  # 根据分类概率计算分类结果
                result_index = result_index.long()
                test_label = label_ori.long()
                for j in range(len(test_label)):
                    if test_label[j] == result_index[j]:
                        confusion_matrix[i][test_label[j]][test_label[j]] += 1
                    else:
                        confusion_matrix[i][test_label[j]][result_index[j]] += 1
    return confusion_matrix


def test_model(model, test_dataloader_list):
    """模型测试"""
    model.eval()

    # 定义统计指标
    confusion_matrix = torch.zeros(2, 2)
    softmax = torch.nn.Softmax(dim=1)  # 计算分类概率，dim=1表示按行计算

    batch_num = int(params.test_num_samples / params.sample_size)
    print('\nbatch_num:{}\n'.format(batch_num))

    # 判断result.txt的目录是否存在，不存在则递归创建目录
    if not os.path.exists(params.result_file_root):
        os.makedirs(params.result_file_root)

    prob_all = []  # 记录预测概率
    label_all = []  # 记录对应的标签
    data_zip = zip(test_dataloader_list[0], test_dataloader_list[1])
    label_list = []
    with torch.no_grad():
        for batch_index, (D1_T_data, D1_F_data) in enumerate(tqdm(data_zip)):

            data_list_ori = [D1_T_data, D1_F_data]
            face_ori, mask_ori, label_ori, Domain_ID_ori, path_ori, input_face_next, input_mask_next = cat_multiDomains(data_list_ori)
            out = model(face_ori, face_ori, input_face_next)
            pred_cls = out[0]
            label_list.append(label_ori)

            prob_all.extend(pred_cls[:, 1].cpu().data)  # prob[:,1]返回每一行第二列的数，roc_auc_score函数需要最大标签类对应的概率
            label_all.extend(label_ori.cpu().data)

            """计算分类结果"""
            preds = softmax(pred_cls)
            result_index = getResult(preds, threshold=0.5)  # 根据分类概率计算分类结果
            result_index = result_index.long()
            test_label = label_ori.long()
            for j in range(len(test_label)):
                if test_label[j] == result_index[j]:
                    confusion_matrix[test_label[j]][test_label[j]] += 1
                else:
                    confusion_matrix[test_label[j]][result_index[j]] += 1

            # 混淆矩阵位置调整
            TP = confusion_matrix[1][1]
            TN = confusion_matrix[0][0]
            FN = confusion_matrix[1][0]
            FP = confusion_matrix[0][1]
            acc_test = float(100.0 * ((TP + TN) / (TP + TN + FN + FP)))
            TPR = 100.0 * (TP / (TP + FN))
            FPR = 100.0 * (FP / (FP + TN))
            TNR = 100 - FPR
            HTER = 100 - (TPR + TNR) / 2

            if ((batch_index + 1) % params.print_freq == 0):
                print('batch[{}/{}]'.format(batch_index + 1, batch_num))
                print('【confusion_matrix】:')
                print(torch.tensor([[TP, FN], [FP, TN]]))
                print('【acc_test:{%f%%}】, 【TPR:{%f%%}】, 【FPR:{%f%%}】, 【TNR:{%f%%}】, 【HTER:{%f%%}】' %
                      (float(acc_test), float(TPR), float(FPR), float(TNR), float(HTER)))
  
    # 混淆矩阵位置调整
    TP = confusion_matrix[1][1]
    TN = confusion_matrix[0][0]
    FN = confusion_matrix[1][0]
    FP = confusion_matrix[0][1]
    acc_test = float(100.0 * ((TP + TN) / (TP + TN + FN + FP)))
    TPR = 100.0 * (TP / (TP + FN))
    FPR = 100.0 * (FP / (FP + TN))
    TNR = 100 - FPR
    HTER = 100 - (TPR + TNR) / 2
    print('【confusion_matrix】:')
    print(torch.tensor([[TP, FN], [FP, TN]]))
    print('【ii:{end}】')
    print('【acc_test:{%f%%}】, 【TPR:{%f%%}】, 【FPR:{%f%%}】, 【TNR:{%f%%}】, 【HTER:{%f%%}】' %
          (float(acc_test), float(TPR), float(FPR), float(TNR), float(HTER)))

    """绘制真假样本预测概率的分布图"""
    pred_T = []
    pred_F = []
    for i in range(len(label_all)):
        if label_all[i] == 1:
            pred_T.append(prob_all[i])
        else:
            pred_F.append(prob_all[i])
    bins = np.arange(-0.1, 1.1, 0.01)  # 设置连续的边界值，即直方图的分布区间[0,10],[10,20]...
    # 直方图会进行统计各个区间的数值,normed=True是频率图，默认是频数图,alpha设置透明度，0为完全透明
    frequency_T, _, _ = plt.hist(pred_T, bins, density=True, stacked=True, facecolor='blue', edgecolor='k', alpha=0.5)
    frequency_F, _, _ = plt.hist(pred_F, bins, density=True, stacked=True, facecolor='red', edgecolor='k', alpha=0.5)
    formatter = FuncFormatter(to_percent)
    plt.gca().yaxis.set_major_formatter(formatter)
    fontsize = 15
    plt.xlabel('pred scores', fontsize=fontsize)
    plt.ylabel('probability', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)  # 设置xy轴刻度文字大小
    plt.yticks(fontsize=fontsize)
    plt.xlim([-0.1, 1.1])
    plt.ylim([0, 100])
    plt.legend(["Real", "Fake"], fontsize=fontsize)
    plt.show()

    AUC = roc_auc_score(label_all, prob_all)  # label_all, prob_all形状要一样
    print("AUC:{:.2f}".format(AUC * 100))
    fpr, tpr, thersholds = roc_curve(label_all, prob_all, pos_label=1)
    AUC2 = auc(fpr, tpr)  # 计算AUC
    print("AUC:{:.2f}".format(AUC2 * 100))
    print(len(thersholds))

    # 计算EER和对应的阈值
    diff = []
    for i, value in enumerate(thersholds):
        far = fpr[i]  # 错误接受率
        frr = 1 - tpr[i]  # 错误拒绝率
        diff_temp = abs(far - frr)
        diff.append(diff_temp)
        # print("%f %f %f" % (fpr[i], tpr[i], value))
    min_diff = min(diff)
    min_loc = diff.index(min_diff)
    far_selected = fpr[min_loc] * 100
    frr_selected = (1 - tpr[min_loc]) * 100
    thershold_selected = thersholds[min_loc]
    hter = 0.5 * (far_selected + frr_selected)
    print("EER: far=%f%%, frr=%f%%, thershold_selected=%f" % (far_selected, frr_selected, thershold_selected))
    print("EER=%f%%, HTER=%f%%" % (hter, hter))

    # 绘制ROC曲线
    plt.figure()
    plt.plot(fpr, tpr, 'k--', label='SPSL (AUC = {0:.2f})'.format(AUC2 * 100), lw=2)
    # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')  # 可以使用中文，但需要导入一些库即字体
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()
    plt.close()

def CAM_test(model, test_dataloader_list):
    """模型测试"""
    import torchvision.transforms.functional as TF
    model = model.module.to("cuda:0")
    model.eval()
    seed = 2026
    # 定义统计指标
    batch_num = int(params.test_num_samples / params.sample_size)
    print('\nbatch_num:{}\n'.format(batch_num))
    source_transforms = get_source_transforms()
    transforms = get_transforms()
    
    cam_save_dir = os.path.join("./data", 'cam')
    
    data_zip = zip(test_dataloader_list[0], test_dataloader_list[1])
    
    # target_layer = model.seg_decoder.decoder[-7]
    target_layer = model.seg_decoder.decoder[-4]
    activations = None
    gradients = None

    def save_activation_hook(module, input, output):
        nonlocal activations
        activations = output.detach()

    def save_gradient_hook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0].detach()
    handle_act = target_layer.register_forward_hook(save_activation_hook)
    handle_grad = target_layer.register_full_backward_hook(save_gradient_hook)
    # 将热力图叠加到原图
    for batch_index, (D1_T_data, D1_F_data) in enumerate(tqdm(data_zip)):
        data_list_ori = [D1_T_data, D1_F_data]
        face_ori, mask_ori, label_ori, Domain_ID_ori, path_ori, input_face_next, input_mask_next = cat_multiDomains(data_list_ori)
        out = model(face_ori.to("cuda:0"), face_ori.to("cuda:0"), input_face_next.to("cuda:0"))
        probs = out[0]           # 分类logits，shape: (batch, num_classes)
        mask = out[1]
        features = out[5]           # 特征图，shape: (batch, nc, h, w)
        out_dir = int(out[6].cpu().numpy())
        # predicted = torch.argmax(probs, dim=1)

        # if out_dir == 0:
        #     out_dir="ff_df"
        # elif out_dir == 1:
        #     out_dir="ff_fs"
        # elif out_dir== 2:
        #     out_dir="ff_f2f"
        # elif out_dir == 3:
        #     out_dir="ff_nt"
        # elif out_dir == 4:
        #     out_dir="ff_fsh"
        if out_dir== 0:
            out_dir="dfd"
        elif out_dir== 1:
            out_dir="dfdc"
        elif out_dir== 2:
            out_dir="dfdc_p"
        elif out_dir== 3:
            out_dir="cdfv1"
        elif out_dir== 4:
            out_dir="cdfv2"
        elif out_dir== 5:
            out_dir="deeper"
            
        score = mask[:, 1, :, :].sum()
        model.zero_grad()
        score.backward()
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        # cam = weights * activations
        cam =(weights * activations).sum(dim=1, keepdim=True)
        
        handle_act.remove()
        handle_grad.remove()
        sigma = 4.0
        k_size = int(sigma * 4) + 1
        if k_size % 2 == 0: k_size += 1
        for i in range(mask.shape[0]):
            cam_i = cam[i,None]
            cam_i = (cam_i - cam_i.min())/(cam_i.max()-cam_i.min())
            # cam_i = F.relu(cam_i)
            # cam_i = (cam_i - cam_i.min())/(cam_i.max()-cam_i.min())
            cam_i = TF.gaussian_blur(cam_i, kernel_size=(k_size, k_size), sigma=(sigma, sigma))
            cam_i = F.interpolate(cam_i, size=(256, 256), mode='bilinear', align_corners=False)
            img_np = face_ori.permute(0, 2, 3, 1).cpu().numpy()[i, :, :, :].astype(np.uint8)
            ma_ori = mask_ori.permute(0, 2, 3, 1).cpu().numpy()[i, :, :, :].astype(np.uint8)
            ma = ((mask > 0.5).float() * 255).permute(0, 2, 3, 1).cpu().numpy()[i, :, :, :].astype(np.uint8)
            heatmap = (cam_i * 255).permute(0, 2, 3, 1).cpu().numpy()[0].astype(np.uint8)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            background = Image.fromarray(img_np)
            foreground = Image.fromarray(heatmap)
            result = Image.blend(foreground, background, alpha=0.7)
            cam_dir = os.path.join(str(cam_save_dir), out_dir)
            if not os.path.exists(cam_dir):
                os.makedirs(cam_dir)
            number = len(os.listdir(cam_dir))
            if i < 2:
                name = "true"
                continue
            else:
                name = "false"
            save_name = str(i) + "_" + str(number) + "_" + name + "_cam.jpg"
            background_save_name = str(i) + "_" + str(number) + "_" + name + "_background.jpg"
            mask_save_name = str(i) + "_" + str(number) + "_" + name + "_mask.jpg"
            mask_pred_save_name = str(i) + "_" + str(number) + "_" + name + "_mask_pred.jpg"
            
            save_path = os.path.join(cam_dir, save_name)
            background_save_path = os.path.join(cam_dir, background_save_name)
            mask_save_path = os.path.join(cam_dir, mask_save_name)
            mask_pred_save_path = os.path.join(cam_dir, mask_pred_save_name)
            result.save(save_path)
            background.save(background_save_path)
            Image.fromarray(np.squeeze(ma_ori*255)).save(mask_save_path)
            Image.fromarray(np.squeeze(ma[:, :, 1])).save(mask_pred_save_path)
            print(save_path)
        
        # if out_dir== 0:
        #     out_dir="dfd"
        # elif out_dir== 1:
        #     out_dir="dfdc"
        # elif out_dir== 2:
        #     out_dir="dfdc_p"
        # elif out_dir== 3:
        #     out_dir="cdfv1"
        # elif out_dir== 4:
        #     out_dir="cdfv2"
        # elif out_dir== 5:
        #     out_dir="deeper"
        # elif out_dir== 6:
        #     out_dir="ff"
        # 计算预测概率和预测类别
        
        # with torch.no_grad():
            
        
        #     # batch_size, nc, h, w = features.shape
        #     # upsample_size = (256, 256)  # 根据实际输入尺寸调整，通常为输入图像大小
        #     # for sample_idx in range(batch_size):
        #     #     # 当前样本的预测类别（也可改为label_ori[sample_idx]使用GT类别）
        #     #     class_idx = predicted[sample_idx].item()
                
        #     #     # 计算单张图像的CAM
        #     #     feature_conv = features[sample_idx]  # (nc, h, w)
        #     #     cam = torch.matmul(weight_softmax[:,class_idx], feature_conv.reshape(nc, h * w))  # (h*w,)
        #     #     cam = cam.reshape(h, w)
                
        #     #     # 归一化到[0, 255]
        #     #     cam = cam - cam.min()
        #     #     cam = cam / (cam.max() + 1e-8)
        #     #     cam_img = (cam.detach().cpu().numpy() * 255).astype(np.uint8)
        #     #     cam_img = cv2.resize(cam_img, upsample_size)
                
        #     #     # 热力图
        #     #     heatmap = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)  # BGR格式
        #     #     heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                
        #     #     img_np = face_ori.permute(0, 2, 3, 1).cpu().numpy()[sample_idx, :, :, :].astype(np.uint8)
        #     #     ma_ori = mask_ori.permute(0, 2, 3, 1).cpu().numpy()[sample_idx, :, :, :].astype(np.uint8)
        #     #     ma = ((mask > 0.5).float() * 255).permute(0, 2, 3, 1).cpu().numpy()[sample_idx, :, :, :].astype(np.uint8)
        #     #     # ma_ori_re = mask_ori_re.permute(0, 2, 3, 1).cpu().numpy()[sample_idx, :, :, :].astype(np.uint8)
        #     #     # heatmap = heatmap * ma_ori
        #     #     # heatmap = heatmap + img_np * ma_ori_re
                
        #     #     # img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                
        #     #     # 叠加（可调整权重）
        #     #     background = Image.fromarray(img_np)
        #     #     foreground = Image.fromarray(heatmap)
        #     #     result = Image.blend(foreground, background, alpha=0.7)
                
        #     #     cam_dir = os.path.join(str(cam_save_dir), out_dir)
        #     #     if not os.path.exists(cam_dir):
        #     #         os.makedirs(cam_dir)
        #     #     number = len(os.listdir(cam_dir))
        #     #     if sample_idx < 2:
        #     #         name = "true"
        #     #         continue
        #     #     else:
        #     #         name = "false"
        #     #     save_name = str(sample_idx) + "_" + str(number) + "_" + name + "_cam.jpg"
        #     #     background_save_name = str(sample_idx) + "_" + str(number) + "_" + name + "_background.jpg"
        #     #     # foreground_save_name = str(sample_idx) + "_" + str(number) + "_" + name + "_foreground.jpg"
        #     #     mask_save_name = str(sample_idx) + "_" + str(number) + "_" + name + "_mask.jpg"
        #     #     mask_pred_save_name = str(sample_idx) + "_" + str(number) + "_" + name + "_mask_pred.jpg"
                
        #     #     save_path = os.path.join(cam_dir, save_name)
        #     #     background_save_path = os.path.join(cam_dir, background_save_name)
        #     #     # foreground_save_path = os.path.join(cam_dir, foreground_save_name)
        #     #     mask_save_path = os.path.join(cam_dir, mask_save_name)
        #     #     mask_pred_save_path = os.path.join(cam_dir, mask_pred_save_name)
        #     #     result.save(save_path)
        #     #     background.save(background_save_path)
        #     #     # foreground.save(foreground_save_path)
        #     #     Image.fromarray(np.squeeze(ma_ori*255)).save(mask_save_path)
        #     #     Image.fromarray(np.squeeze(ma[:, :, 1])).save(mask_pred_save_path)
        #     #     print(save_path)
        #     mask_ori_re = torch.where(mask_ori == 0, 1, 0)
        #     mask_ones = torch.ones(mask_ori.shape).to(mask_ori.device) * 255
        #     out = imageRelighting_plot(
        #         input_image=face_ori,
        #         input_mask=mask_ones,
        #         reference_image=face_ori,
        #         batch_size=face_ori.shape[0],
        #         devices="cuda:0")
        #     Domains_data_list = data_list_ori
        #     Domain_number = len(Domains_data_list)
        #     input_data = Domains_data_list[0]
        #     rel_landmarks = input_data[7]
        #     for i in range(Domain_number - 1):
        #         input_data2 = Domains_data_list[i + 1]
        #         rel_landmarks2 = input_data2[7]
        #         rel_landmarks = np.concatenate((rel_landmarks, rel_landmarks2), axis=0)

        #     for data in out:
        #         face_tar, shadow_mask, albedo, depth, shading, surface_normals = data["data"]
        #         direction = data["direction"]
        #         Domain_number = len(Domains_data_list)

                
        #         for ide, i in enumerate(range(face_ori.shape[0])):
        #             if i < 2:
        #                 name = "true"
        #             else:
        #                 name = "false"
                    
        #             ori = face_ori.permute(0, 2, 3, 1).cpu().numpy()[i, :, :, :].astype(np.uint8)
        #             ma_ori = mask_ori.permute(0, 2, 3, 1).cpu().numpy()[i, :, :, :].astype(np.uint8)
        #             tar = face_tar.permute(0, 2, 3, 1).cpu().numpy()[i, :, :, :].astype(np.uint8)
        #             sh_mask = shadow_mask.permute(0, 2, 3, 1).cpu().numpy()[i, :, :, :].astype(np.uint8)
        #             al = albedo.permute(0, 2, 3, 1).cpu().numpy()[i, :, :, :].astype(np.uint8)
        #             de = depth.permute(0, 2, 3, 1).cpu().numpy()[i, :, :, :].astype(np.uint8)
        #             shad = shading.permute(0, 2, 3, 1).cpu().numpy()[i, :, :, :].astype(np.uint8)
        #             surface = surface_normals.permute(0, 2, 3, 1).cpu().numpy()[i, :, :, :].astype(np.uint8)
        #             ma_ori_re = mask_ori_re.permute(0, 2, 3, 1).cpu().numpy()[i, :, :, :].astype(np.uint8)

        #             tar_mask = tar * ma_ori
        #             ori_reshape = tar_mask + ori * ma_ori_re

        #             # sbi生成
        #             img_face_sbi_false_list = []
        #             for j in range(10):
        #                 current_seed = seed + j 
        #                 np.random.seed(current_seed)
        #                 random.seed(current_seed)
        #                 img_face_sbi_true, img_face_sbi_false, mask_face_sbi = self_blending_ori(
        #                     ori_reshape, rel_landmarks[i], source_transforms, j
        #                 )
        #                 if j%2 == 0:
        #                     transformed=transforms(image=img_face_sbi_false.astype('uint8'),image1=img_face_sbi_true.astype('uint8'))
        #                     transformed = transformed['image']
        #                 else:
        #                     transformed = img_face_sbi_false
        #                 img_face_sbi_false_list.append(transformed)

        #             images = [ori, ma_ori, tar, sh_mask, al, de, shad, surface, ori_reshape, img_face_sbi_false_list[0]]
        #             titles = ["Original Face", "Original Mask", "Target Face", "Shadow Mask", "Albedo",
        #                     "Depth", "Shading", "Surface Normals", "Original Reshape", "Original Face SBI"]
        #             fig = plt.figure(figsize=(16, 10))
        #             grid = plt.GridSpec(2, 5, wspace=0.3, hspace=0.3)
        #             positions = [
        #                 grid[0, 0], grid[0, 1], grid[0, 2], grid[0, 3], grid[0, 4],
        #                 grid[1, 0], grid[1, 1], grid[1, 2], grid[1, 3], grid[1, 4]
        #             ]
        #             for idx, (img, title, pos) in enumerate(zip(images, titles, positions)):
        #                 ax = fig.add_subplot(pos)
        #                 if img.shape[2] == 3:
        #                     ax.imshow(img)
        #                 elif img.shape[2] == 1:
        #                     ax.imshow(img[:, :, 0], cmap="gray")
        #                 ax.set_title(title, fontsize=12, fontweight="bold")
        #                 ax.axis("off")
        #             ax = fig.add_subplot(grid[1, 4])
        #             ax.axis("off")
        #             save_dir = os.path.join("./data/relight/" + direction, out_dir + "_" + name)
        #             if not os.path.exists(save_dir):
        #                 os.makedirs(save_dir)
        #             number = len(os.listdir(save_dir))
        #             save_dir = os.path.join(save_dir, str(i) + "_" + str(number))
        #             plt.savefig(save_dir + '_all.jpg')
        #             ma_ori = np.squeeze(ma_ori)
        #             sh_mask = np.squeeze(ma_ori)
        #             shad = np.squeeze(shad)
        #             de = np.squeeze(de)
                    
        #             Image.fromarray(ori).save(save_dir + "_ori.jpg")
        #             Image.fromarray(ma_ori * 255).save(save_dir + "_ori_mask.jpg")
        #             Image.fromarray(tar).save(save_dir + "_tar.jpg")
        #             Image.fromarray(sh_mask).save(save_dir + "_shadow_mask.jpg")
        #             Image.fromarray(al).save(save_dir + "_albedo.jpg")
        #             Image.fromarray(de).save(save_dir + "_depth.jpg")
        #             Image.fromarray(shad).save(save_dir + "_shading.jpg")
        #             Image.fromarray(surface).save(save_dir + "_surface_normals.jpg")
        #             Image.fromarray(ori_reshape).save(save_dir + "_ori_reshape.jpg")
        #             for x in range(10):
        #                 Image.fromarray(img_face_sbi_false_list[x]).save(save_dir + f"_ori_sbi_{x}.jpg")
        #             print(f"out {save_dir}" + "_all.jpg")
                

def self_blending(img, landmark, replay_data=None):
        # 生成原始 mask
        mask = np.zeros_like(img[:, :, 0]).astype(np.float32)
        cv2.fillConvexPoly(mask, cv2.convexHull(landmark.astype(np.int32)), 1.)
        
        source = img.copy()
        source_transforms = A.ReplayCompose([
            A.Compose([
                A.RGBShift((-20, 20), (-20, 20), (-20, 20), p=0.3),
                A.HueSaturationValue(hue_shift_limit=(-0.3, 0.3), sat_shift_limit=(-0.3, 0.3),
                                     val_shift_limit=(-0.3, 0.3), p=1),
                A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=1),
            ], p=1),
            A.OneOf([
                # 如果没有 RandomDownScale，可以用 A.Downscale
                A.Downscale(scale_min=0.5, scale_max=0.5, p=1),
                A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1),
            ], p=1),
        ], p=1.)
        if replay_data is None:
            # 第一张图：正常增强并记录参数
            if np.random.rand() < 0.5:
                # 记录 source 的增强
                res = source_transforms(image=source.astype(np.uint8))
                source = res['image']
                applied_replay = res['replay'] # 获取本次增强的参数
                aug_target = 'source'
            else:
                # 记录 img 的增强
                res = source_transforms(image=img.astype(np.uint8))
                img = res['image']
                applied_replay = res['replay']
                aug_target = 'img'
        else:
            # 第二张图：使用传入的 replay_data 复现参数
            applied_replay = replay_data['replay']
            aug_target = replay_data['target']
            if aug_target == 'source':
                source = A.ReplayCompose.replay(applied_replay, image=source.astype(np.uint8))['image']
            else:
                img = A.ReplayCompose.replay(applied_replay, image=img.astype(np.uint8))['image']

        # 仿射变换部分（如果也需要同步，建议对 randaffine 也做类似处理）
        source, mask = randaffine(source, mask)
        mask = np.expand_dims(mask, axis=-1)
        img_blended = (source * mask + img * (1 - mask)).astype(np.uint8)

        # 返回结果时带上增强参数，供下一张图使用
        return img.astype(np.uint8), img_blended.astype(np.uint8), mask.squeeze(), {"replay": applied_replay, "target": aug_target}

class RandomDownScale(A.core.transforms_interface.ImageOnlyTransform):
    def apply(self,img,**params):
        return self.randomdownscale(img)

    def randomdownscale(self,img):
        keep_ratio=True
        keep_input_shape=True
        H,W,C=img.shape
        ratio_list=[2,4]
        r=ratio_list[np.random.randint(len(ratio_list))]
        img_ds=cv2.resize(img,(int(W/r),int(H/r)),interpolation=cv2.INTER_NEAREST)
        if keep_input_shape:
            img_ds=cv2.resize(img_ds,(W,H),interpolation=cv2.INTER_LINEAR)

        return img_ds

def get_source_transforms():
		return A.Compose([
				A.Compose([
						A.RGBShift((-20,20),(-20,20),(-20,20),p=0.3),
						A.HueSaturationValue(hue_shift_limit=(-0.3,0.3), sat_shift_limit=(-0.3,0.3), val_shift_limit=(-0.3,0.3), p=1),
						A.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1,0.1), p=1),
					],p=1),
	
				A.OneOf([
					RandomDownScale(p=1),
					A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1),
				],p=1),
				
			], p=1.)
		
def get_transforms():
    return A.Compose([
        A.RGBShift((-20,20),(-20,20),(-20,20),p=0.3),
        A.HueSaturationValue(hue_shift_limit=(-0.3,0.3), sat_shift_limit=(-0.3,0.3), val_shift_limit=(-0.3,0.3), p=0.3),
        A.RandomBrightnessContrast(brightness_limit=(-0.3,0.3), contrast_limit=(-0.3,0.3), p=0.3),
        A.ImageCompression(quality_lower=40,quality_upper=100,p=0.5),
    ], 
    additional_targets={f'image1': 'image'},
    p=1.)

def self_blending_ori(img,landmark, source_transforms, number):
    H,W=len(img),len(img[0])
    if np.random.rand()<0.25:
        landmark=landmark[:68]
    mask=np.zeros_like(img[:,:,0])
    cv2.fillConvexPoly(mask, cv2.convexHull(landmark), 1.)


    source = img.copy()
    if np.random.rand()<0.5:
        source = source_transforms(image=source.astype(np.uint8))['image']
    else:
        img = source_transforms(image=img.astype(np.uint8))['image']

    source, mask = randaffine(source,mask)

    img_blended,mask=dynamic_blend(source,img,mask)
    img_blended = img_blended.astype(np.uint8)
    img = img.astype(np.uint8)

    return img,img_blended,mask
	
def dynamic_blend(source,target,mask):
    mask_blured = get_blend_mask(mask)
    blend_list=[0.25,0.5,0.75,1,1,1]
    blend_ratio = blend_list[np.random.randint(len(blend_list))]
    mask_blured*=blend_ratio
    img_blended=(mask_blured * source + (1 - mask_blured) * target)
    return img_blended,mask_blured

def get_blend_mask(mask):
	H,W=mask.shape
	size_h=np.random.randint(192,257)
	size_w=np.random.randint(192,257)
	mask=cv2.resize(mask,(size_w,size_h))
	kernel_1=random.randrange(5,26,2)
	kernel_1=(kernel_1,kernel_1)
	kernel_2=random.randrange(5,26,2)
	kernel_2=(kernel_2,kernel_2)
	
	mask_blured = cv2.GaussianBlur(mask, kernel_1, 0)
	mask_blured = mask_blured/(mask_blured.max())
	mask_blured[mask_blured<1]=0
	
	mask_blured = cv2.GaussianBlur(mask_blured, kernel_2, np.random.randint(5,46))
	mask_blured = mask_blured/(mask_blured.max())
	mask_blured = cv2.resize(mask_blured,(W,H))
	return mask_blured.reshape((mask_blured.shape+(1,)))

def randaffine(img, mask):
        f = A.Affine(
            translate_percent={'x': (-0.03, 0.03), 'y': (-0.015, 0.015)},
            scale=[0.95, 1 / 0.95],
            fit_output=False,
            p=1)

        g = A.ElasticTransform(
            alpha=50,
            sigma=7,
            alpha_affine=0,
            p=1,
        )
        transformed = f(image=img, mask=mask)
        img = transformed['image']
        mask = transformed['mask']
        transformed = g(image=img, mask=mask)
        mask = transformed['mask']
        return img, mask

"""特征分布图可视化相关函数============================================================================================="""

def plot_embedding(data, label, title):
    """
    :param data: 数据集 (n,2) 二维数组（t-SNE降维后）
    :param label: 样本标签 (n,) 一维数组
    :param title: 图像标题
    :return: 图像fig实例
    """
    # 归一化处理（保留原逻辑）
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min) if (x_max - x_min).sum() != 0 else data
    
    fig = plt.figure(figsize=(8, 8))  # 新增：指定画布大小，避免过小
    ax = fig.add_subplot(111)         # 更规范的子图创建方式，替代plt.subplot(111)
    
    # 核心优化：按标签分组绘制散点（一次绘一个标签，替代逐点循环，无渲染隐患）
    colors = ['red', 'blue', 'green', 'purple', 'orange']  # 标签0-4对应颜色
    markers = 'o'
    s = 15
    for lab in range(5):  # 遍历标签0-4
        # 筛选当前标签的所有数据点
        idx = label == lab
        if np.any(idx):  # 只有该标签有数据时才绘制，避免空数据报错
            ax.scatter(data[idx, 0], data[idx, 1], 
                       color=colors[lab], marker=markers, s=s, 
                       label=f'label_{lab}', edgecolors='none')  # 无边框，更清晰
    
    # 隐藏坐标轴（保留原逻辑，规范操作）
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    return fig


def plot_embedding3(data, label, title):
    """
    :param data:数据集
    :param label:样本标签
    :param title:图像标题
    :return:图像
    """
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)  # 对数据进行归一化处理
    fig = plt.figure()  # 创建图形实例
    ax = plt.subplot(111)  # 创建子图

    fontsize = 50
    alpha = 0.5
    # 遍历所有样本
    for i in range(data.shape[0]):
        # 在图中为每个数据点画出标签
        if label[i] == 0:
            colorplt = 'blue'
            scatter0 = plt.scatter(data[i, 0], data[i, 1], color=colorplt, marker='o', s=fontsize, alpha=alpha)
        elif label[i] == 1:
            colorplt = 'blue'
            scatter1 = plt.scatter(data[i, 0], data[i, 1], color=colorplt, marker='o', s=fontsize, alpha=alpha)
        elif label[i] == 2:
            colorplt = 'red'
            scatter2 = plt.scatter(data[i, 0], data[i, 1], color=colorplt, marker='+', s=fontsize, alpha=alpha)
        elif label[i] == 3:
            colorplt = 'red'
            scatter3 = plt.scatter(data[i, 0], data[i, 1], color=colorplt, marker='+', s=fontsize, alpha=alpha)
        elif label[i] == 4:
            colorplt = 'green'
            scatter4 = plt.scatter(data[i, 0], data[i, 1], color=colorplt, marker='^', s=fontsize, alpha=alpha)
        elif label[i] == 5:
            colorplt = 'green'
            scatter5 = plt.scatter(data[i, 0], data[i, 1], color=colorplt, marker='^', s=fontsize, alpha=alpha)
        elif label[i] == 6:
            colorplt = 'purple'
            scatter6 = plt.scatter(data[i, 0], data[i, 1], color=colorplt, marker='s', s=fontsize, alpha=alpha)
        elif label[i] == 7:
            colorplt = 'purple'
            scatter7 = plt.scatter(data[i, 0], data[i, 1], color=colorplt, marker='s', s=fontsize, alpha=alpha)
        elif label[i] == 8:
            colorplt = 'orange'
            scatter8 = plt.scatter(data[i, 0], data[i, 1], color=colorplt, marker='D', s=fontsize, alpha=alpha)
        elif label[i] == 9:
            colorplt = 'orange'
            scatter9 = plt.scatter(data[i, 0], data[i, 1], color=colorplt, marker='D', s=fontsize, alpha=alpha)
        if i % 200 == 0:
            print('Processing {}-th samples'.format(i))

    font = {'family': 'Times New Roman', 'size': 15, }
    plt.legend((scatter1, scatter0, scatter3, scatter2, scatter5, scatter4, scatter7, scatter6, scatter9, scatter8),
               ('FF++/Real', 'FF++/Fake', 'DFD/Real', 'DFD/Fake', 'DFDC/Real', 'DFDC/Fake', 'CDF/Real', 'CDF/Fake',
                'Deeper/Real', 'Deeper/Fake'),
               loc='upper right', fontsize=15, prop=font)
    # plt.legend((scatter1, scatter0, scatter3, scatter2, scatter5, scatter4, scatter7, scatter6),
    #            ('FF++/Real', 'FF++/Fake', 'DFD/Real', 'DFD/Fake', 'DFDC/Fake', 'DFDC/Real', 'CDF/Fake', 'CDF/Real'),
    #            loc='upper right', fontsize=15, prop=font)
    plt.xlim(-0.5, 1.5)
    plt.ylim(-0.5, 1.5)
    plt.axes().get_xaxis().set_visible(False)  # 隐藏x坐标轴
    plt.axes().get_yaxis().set_visible(False)  # 隐藏y坐标轴
    return fig


def plot_embedding3_v0(data, label, title):
    """
    :param data:数据集
    :param label:样本标签
    :param title:图像标题
    :return:图像
    """
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)  # 对数据进行归一化处理
    fig = plt.figure()  # 创建图形实例
    ax = plt.subplot(111)  # 创建子图

    fontsize = 50
    # fontsize = 80
    alpha = 0.5
    # 遍历所有样本
    for i in range(data.shape[0]):
        # 在图中为每个数据点画出标签
        if label[i] == 0:
            colorplt = 'red'
            scatter0 = plt.scatter(data[i, 0], data[i, 1], color=colorplt, marker='o', s=fontsize, alpha=alpha)
        elif label[i] == 1:
            colorplt = 'blue'
            scatter1 = plt.scatter(data[i, 0], data[i, 1], color=colorplt, marker='o', s=fontsize, alpha=alpha)

        elif label[i] == 2:
            colorplt = 'red'
            scatter2 = plt.scatter(data[i, 0], data[i, 1], color=colorplt, marker='+', s=fontsize, alpha=alpha)
        elif label[i] == 3:
            colorplt = 'blue'
            scatter3 = plt.scatter(data[i, 0], data[i, 1], color=colorplt, marker='+', s=fontsize, alpha=alpha)

        elif label[i] == 4:
            colorplt = 'red'
            scatter4 = plt.scatter(data[i, 0], data[i, 1], color=colorplt, marker='^', s=fontsize, alpha=alpha)
        elif label[i] == 5:
            colorplt = 'blue'
            scatter5 = plt.scatter(data[i, 0], data[i, 1], color=colorplt, marker='^', s=fontsize, alpha=alpha)

        elif label[i] == 6:
            colorplt = 'red'
            scatter6 = plt.scatter(data[i, 0], data[i, 1], color=colorplt, marker='s', s=fontsize, alpha=alpha)
        elif label[i] == 7:
            colorplt = 'blue'
            scatter7 = plt.scatter(data[i, 0], data[i, 1], color=colorplt, marker='s', s=fontsize, alpha=alpha)

        elif label[i] == 8:
            colorplt = 'red'
            scatter8 = plt.scatter(data[i, 0], data[i, 1], color=colorplt, marker='D', s=fontsize, alpha=alpha)
        elif label[i] == 9:
            colorplt = 'blue'
            scatter9 = plt.scatter(data[i, 0], data[i, 1], color=colorplt, marker='D', s=fontsize, alpha=alpha)
        if i % 200 == 0:
            print('Processing {}-th samples'.format(i))

    font = {'family': 'Times New Roman', 'size': 15, }
    plt.legend((scatter1, scatter0, scatter3, scatter2, scatter5, scatter4, scatter7, scatter6, scatter9, scatter8),
               ('FF++/Real', 'FF++/Fake', 'DFD/Real', 'DFD/Fake', 'DFDC/Real', 'DFDC/Fake', 'CDF/Real', 'CDF/Fake',
                'Deeper/Real', 'Deeper/Fake'),
               loc='upper right', fontsize=15, prop=font)
    plt.xlim(-0.5, 1.5)
    plt.ylim(-0.5, 1.5)
    plt.axes().get_xaxis().set_visible(False)  # 隐藏x坐标轴
    plt.axes().get_yaxis().set_visible(False)  # 隐藏y坐标轴
    return fig


def plot_embedding3_v2_new(data, label, title):
    """
    添加shift变量,计算数据集的style分布+增强的风格分布
    :param data:数据集
    :param label:样本标签
    :param title:图像标题
    :return:图像
    """
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)  # 对数据进行归一化处理
    fig = plt.figure()  # 创建图形实例
    ax = plt.subplot(111)  # 创建子图

    fontsize = 50
    alpha = 0.5
    shift = 0.5
    gap = 2
    The = 0
    # The = 99999999

    ublabel = -1
    # 遍历所有样本
    for i in range(data.shape[0]):
        # 在图中为每个数据点画出标签
        if label[i] == 0:
            if i % gap == The:
                # if random.randint(1, 2) % 2 == 0:
                if random.randint(1, 2) % 2 == 0:
                    colorplt = 'fuchsia'
                    scatter10 = plt.scatter(data[i, 0] + 0.2 * shift, data[i, 1] - 0.3 * shift, color=colorplt,
                                            marker='+', s=fontsize, alpha=alpha)
                    scatter10 = plt.scatter(data[i, 0] + 0.2 * shift, data[i, 1] - 0.9 * shift, color=colorplt,
                                            marker='+', s=fontsize, alpha=alpha)
            colorplt = 'red'
            scatter0 = plt.scatter(data[i, 0], data[i, 1] - 0.6 * shift, color=colorplt, marker='o', s=fontsize,
                                   alpha=alpha)
        elif label[i] == 1:
            colorplt = 'blue'
            scatter1 = plt.scatter(data[i, 0], data[i, 1] + 0.5 * shift, color=colorplt, marker='o', s=fontsize,
                                   alpha=alpha)

        elif label[i] == ublabel:
            if i % gap == The:
                if random.randint(1, 2) % 2 == 0:
                    if random.randint(1, 2) % 2 == 0:
                        colorplt = 'cyan'
                        scatter10 = plt.scatter(data[i, 0] - 1.5 * shift + 0.3 * shift,
                                                data[i, 1] + 1.0 * shift - 0.2 * shift, color=colorplt, marker='+',
                                                s=fontsize, alpha=alpha)
            colorplt = 'red'
            scatter2 = plt.scatter(data[i, 0] - 1.2 * shift + 0.3 * shift, data[i, 1] + 0.5 * shift - 0.2 * shift,
                                   color=colorplt, marker='+', s=fontsize, alpha=alpha)
        elif label[i] == ublabel:
            colorplt = 'blue'
            scatter3 = plt.scatter(data[i, 0] - shift + 0.4 * shift, data[i, 1] + 1.5 * shift - 0.2 * shift,
                                   color=colorplt, marker='+', s=fontsize, alpha=alpha)

        elif label[i] == 4:
            if i % gap == The:
                # if random.randint(1, 2) % 2 == 0:
                if random.randint(1, 2) % 2 == 0:
                    colorplt = 'fuchsia'
                    scatter10 = plt.scatter(data[i, 0] - 0.2 * shift + 1.2 * shift,
                                            data[i, 1] + 1.7 * shift - 0.8 * shift, color=colorplt, marker='+',
                                            s=fontsize, alpha=alpha)
                    scatter10 = plt.scatter(data[i, 0] - 0.2 * shift + 0.2 * shift,
                                            data[i, 1] + 0.5 * shift - 0.8 * shift, color=colorplt, marker='+',
                                            s=fontsize, alpha=alpha)
            colorplt = 'coral'
            scatter4 = plt.scatter(data[i, 0] - 0.2 * shift + 0.8 * shift, data[i, 1] + 1.0 * shift - 0.8 * shift,
                                   color=colorplt, marker='^', s=fontsize, alpha=alpha)
        elif label[i] == 5:
            colorplt = 'dodgerblue'
            scatter5 = plt.scatter(data[i, 0] - 0.2 * shift + 0.6 * shift, data[i, 1] + 1.5 * shift - 0.8 * shift,
                                   color=colorplt, marker='^', s=fontsize, alpha=alpha)

        elif label[i] == ublabel:
            if i % gap == The:
                if random.randint(1, 2) % 2 == 0:
                    if random.randint(1, 2) % 2 == 0:
                        colorplt = 'cyan'
                        scatter10 = plt.scatter(data[i, 0] + 0.6 * shift, data[i, 1] - 0.3 * shift, color=colorplt,
                                                marker='+', s=fontsize, alpha=alpha)
            colorplt = 'red'
            scatter6 = plt.scatter(data[i, 0] + 0.5 * shift, data[i, 1], color=colorplt, marker='s', s=fontsize,
                                   alpha=alpha)
        elif label[i] == ublabel:
            colorplt = 'blue'
            scatter7 = plt.scatter(data[i, 0] + 0.4 * shift, data[i, 1] + shift, color=colorplt, marker='s', s=fontsize,
                                   alpha=alpha)

        elif label[i] == ublabel:
            if i % gap == The:
                if random.randint(1, 2) % 2 == 0:
                    if random.randint(1, 2) % 2 == 0:
                        if random.randint(1, 2) % 2 == 0:
                            colorplt = 'cyan'
                            scatter10 = plt.scatter(data[i, 0] - 1.5 * shift, data[i, 1] - 0.3 * shift, color=colorplt,
                                                    marker='+', s=fontsize, alpha=alpha)
            colorplt = 'red'
            scatter8 = plt.scatter(data[i, 0] - shift, data[i, 1], color=colorplt, marker='D', s=fontsize, alpha=alpha)
        elif label[i] == ublabel:
            colorplt = 'blue'
            scatter9 = plt.scatter(data[i, 0] - 0.5 * shift, data[i, 1] + shift, color=colorplt, marker='D', s=fontsize,
                                   alpha=alpha)
        if i % 200 == 0:
            print('Processing {}-th samples'.format(i))

    font = {'family': 'Times New Roman', 'size': 15, }
    plt.legend((scatter1, scatter0, scatter5, scatter4, scatter10),
               ('FF++/Real', 'FF++/Fake', 'DFDC/Real', 'DFDC/Fake', 'SFFS(Fake)'),
               loc='upper right', fontsize=15, prop=font)
    # plt.xlim(-1.0, 2.0)
    # plt.ylim(-1.0, 2.5)
    plt.xlim(-0.8, 2.0)
    plt.ylim(-0.6, 2.4)
    plt.axes().get_xaxis().set_visible(False)  # 隐藏x坐标轴
    plt.axes().get_yaxis().set_visible(False)  # 隐藏y坐标轴
    return fig


def test_model_TSNE(model, test_dataloader_list):
    """模型测试"""
    model.eval()

    # 定义统计指标
    confusion_matrix = torch.zeros(2, 2)
    softmax = torch.nn.Softmax(dim=1)  # 计算分类概率，dim=1表示按行计算

    batch_num = int(params.test_num_samples / params.sample_size)
    print('\nbatch_num:{}\n'.format(batch_num))

    # 判断result.txt的目录是否存在，不存在则递归创建目录
    if not os.path.exists(params.result_file_root):
        os.makedirs(params.result_file_root)

    fea_list = []
    label_list = []
    prob_all = []  # 记录预测概率
    label_all = []  # 记录对应的标签
    data_zip = zip(test_dataloader_list[0], test_dataloader_list[1])
    with torch.no_grad():
        for batch_index, (D1_T_data, D1_F_data) in enumerate(tqdm(data_zip)):

            data_list = [D1_T_data, D1_F_data]
            # 合并数据
            Ds_face, Ds_mask, Ds_label, Ds_Domain_ID, Ds_path, input_face_next, input_mask_next = cat_multiDomains(data_list)
            # Dt_face, Dt_mask, Dt_label, Dt_Domain_ID, Ds_path = cat_multiDomains(data_list)

            # 复制得到目标域Dt
            Dt_face = Ds_face
            Dt_mask = Ds_mask
            Dt_label = Ds_label
            Dt_Domain_ID = Ds_Domain_ID

            # 模型输出
            # pred_cls, f_rec, f_SR_anchor, f_SR_pairs, pred_dis_invariant = model(Ds_face, Dt_face)
            # pred_cls, pred_cls2, f_SR_anchor, f_SR_pairs, fea_IIE_ori, fea_IIE_rand, pred_dis_invariant, separate_feature, img_separation = model(Ds_face, Dt_face, input_face_next)
            out = model(Ds_face, Dt_face, input_face_next)
            pred_cls = out[0]
            f_SR_anchor = out[4]
            # pred_cls, f_SR_anchor, f_SR_pairs, pred_dis_invariant = model(Ds_face, Dt_face)
            prob_all.extend(pred_cls[:, 1].cpu().data)  # prob[:,1]返回每一行第二列的数，roc_auc_score函数需要最大标签类对应的概率
            label_all.extend(Ds_label.cpu().data)
            f_SR_anchor = f_SR_anchor.reshape(f_SR_anchor.shape[0], -1)
            if batch_index == 0:
                fea_list = f_SR_anchor.cpu().data
                label_list = Ds_label.cpu().data
            else:
                fea_list = torch.cat((fea_list, f_SR_anchor.cpu().data), dim=0)
                label_list = torch.cat((label_list, Ds_label.cpu().data), dim=0)

            """计算分类结果"""
            preds = softmax(pred_cls)
            result_index = getResult(preds, threshold=0.5)  # 根据分类概率计算分类结果
            result_index = result_index.long()
            test_label = Ds_label.long()
            for j in range(len(test_label)):
                if test_label[j] == result_index[j]:
                    confusion_matrix[test_label[j]][test_label[j]] += 1
                else:
                    confusion_matrix[test_label[j]][result_index[j]] += 1

            # 混淆矩阵位置调整
            TP = confusion_matrix[1][1]
            TN = confusion_matrix[0][0]
            FN = confusion_matrix[1][0]
            FP = confusion_matrix[0][1]
            acc_test = float(100.0 * ((TP + TN) / (TP + TN + FN + FP)))
            TPR = 100.0 * (TP / (TP + FN))
            FPR = 100.0 * (FP / (FP + TN))
            TNR = 100 - FPR
            HTER = 100 - (TPR + TNR) / 2

            if ((batch_index + 1) % params.print_freq == 0):
                print('batch[{}/{}]'.format(batch_index + 1, batch_num))
                print('【confusion_matrix】:')
                print(torch.tensor([[TP, FN], [FP, TN]]))
                print('【acc_test:{%f%%}】, 【TPR:{%f%%}】, 【FPR:{%f%%}】, 【TNR:{%f%%}】, 【HTER:{%f%%}】' %
                      (float(acc_test), float(TPR), float(FPR), float(TNR), float(HTER)))

    """绘制T-SNE特征分布可视化"""
    # 定义T-SNE可视化工具。n_components表示降维后嵌入空间的维度，如2或3；init表示嵌入的初始化，可选'pca'或'random'；random_state表示伪随机数发生器种子控制
    ts = manifold.TSNE(n_components=2, init='pca', random_state=0)
    fea_list = fea_list.numpy()
    label_list = label_list.numpy()
    result = ts.fit_transform(fea_list)
    fig = plot_embedding(result, label_list, 't-SNE Embedding')
    number = len(os.listdir("./data/tsne"))
    np.save(os.path.join("./data/tsne",str(number) + "_fea.npy"), fea_list)
    np.save(os.path.join("./data/tsne",str(number) + "_label.npy"), label_list)
    fig.savefig(
        f"./data/tsne/{number}.png",  # 换png，避免jpg有损压缩
        dpi=300,                     # 高清分辨率，散点更清晰
        bbox_inches='tight',         # 必加：去除多余白边，散点显示在图片中央
        pad_inches=0.1,              # 轻微内边距，避免散点贴边（0也可以）
        facecolor='white',           # 画布背景白（默认，可改black）
        edgecolor='none'             # 无边框
    )
    plt.close(fig)

    # 混淆矩阵位置调整
    TP = confusion_matrix[1][1]
    TN = confusion_matrix[0][0]
    FN = confusion_matrix[1][0]
    FP = confusion_matrix[0][1]
    acc_test = float(100.0 * ((TP + TN) / (TP + TN + FN + FP)))
    TPR = 100.0 * (TP / (TP + FN))
    FPR = 100.0 * (FP / (FP + TN))
    TNR = 100 - FPR
    HTER = 100 - (TPR + TNR) / 2
    print('【confusion_matrix】:')
    print(torch.tensor([[TP, FN], [FP, TN]]))
    print('【ii:{end}】')
    print('【acc_test:{%f%%}】, 【TPR:{%f%%}】, 【FPR:{%f%%}】, 【TNR:{%f%%}】, 【HTER:{%f%%}】' %
          (float(acc_test), float(TPR), float(FPR), float(TNR), float(HTER)))

    """绘制真假样本预测概率的分布图"""
    pred_T = []
    pred_F = []
    for i in range(len(label_all)):
        if label_all[i] == 1:
            pred_T.append(prob_all[i])
        else:
            pred_F.append(prob_all[i])
    bins = np.arange(-0.1, 1.1, 0.01)  # 设置连续的边界值，即直方图的分布区间[0,10],[10,20]...
    # 直方图会进行统计各个区间的数值,normed=True是频率图，默认是频数图,alpha设置透明度，0为完全透明
    frequency_T, _, _ = plt.hist(pred_T, bins, density=True, stacked=True, facecolor='blue', edgecolor='k', alpha=0.5)
    frequency_F, _, _ = plt.hist(pred_F, bins, density=True, stacked=True, facecolor='red', edgecolor='k', alpha=0.5)
    formatter = FuncFormatter(to_percent)
    plt.gca().yaxis.set_major_formatter(formatter)
    fontsize = 15
    plt.xlabel('pred scores', fontsize=fontsize)
    plt.ylabel('probability', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)  # 设置xy轴刻度文字大小
    plt.yticks(fontsize=fontsize)
    plt.xlim([-0.1, 1.1])
    plt.ylim([0, 100])
    plt.legend(["Real", "Fake"], fontsize=fontsize)
    plt.show()

    AUC = roc_auc_score(label_all, prob_all)  # label_all, prob_all形状要一样
    print("AUC:{:.2f}".format(AUC * 100))
    fpr, tpr, thersholds = roc_curve(label_all, prob_all, pos_label=1)
    AUC2 = auc(fpr, tpr)  # 计算AUC
    print("AUC:{:.2f}".format(AUC2 * 100))
    print(len(thersholds))

    # 计算EER和对应的阈值
    diff = []
    for i, value in enumerate(thersholds):
        far = fpr[i]  # 错误接受率
        frr = 1 - tpr[i]  # 错误拒绝率
        diff_temp = abs(far - frr)
        diff.append(diff_temp)
        # print("%f %f %f" % (fpr[i], tpr[i], value))
    min_diff = min(diff)
    min_loc = diff.index(min_diff)
    far_selected = fpr[min_loc] * 100
    frr_selected = (1 - tpr[min_loc]) * 100
    thershold_selected = thersholds[min_loc]
    hter = 0.5 * (far_selected + frr_selected)
    print("EER: far=%f%%, frr=%f%%, thershold_selected=%f" % (far_selected, frr_selected, thershold_selected))
    print("EER=%f%%, HTER=%f%%" % (hter, hter))

    # 绘制ROC曲线
    plt.plot(fpr, tpr, 'k--', label='SPSL (AUC = {0:.2f})'.format(AUC2 * 100), lw=2)
    # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')  # 可以使用中文，但需要导入一些库即字体
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

def paper_test_model_TSNE(model, test_dataloader_list):
    """模型测试"""
    model.eval()

    # 定义统计指标
    confusion_matrix = torch.zeros(2, 2)
    softmax = torch.nn.Softmax(dim=1)  # 计算分类概率，dim=1表示按行计算

    batch_num = int(params.test_num_samples / params.sample_size)
    print('\nbatch_num:{}\n'.format(batch_num))

    # 判断result.txt的目录是否存在，不存在则递归创建目录
    if not os.path.exists(params.result_file_root):
        os.makedirs(params.result_file_root)

    fea_list = []
    label_list = []
    prob_all = []  # 记录预测概率
    label_all = []  # 记录对应的标签
    data_zip = zip(test_dataloader_list[0], test_dataloader_list[1])
    with torch.no_grad():
        for batch_index, (D1_T_data, D1_F_data) in enumerate(tqdm(data_zip)):

            data_list = [D1_T_data, D1_F_data]
            # 合并数据
            Ds_face, Ds_mask, Ds_label, Ds_Domain_ID, Ds_path, input_face_next, input_mask_next = cat_multiDomains(data_list)
            # Dt_face, Dt_mask, Dt_label, Dt_Domain_ID, Ds_path = cat_multiDomains(data_list)

            # 复制得到目标域Dt
            Dt_face = Ds_face
            Dt_mask = Ds_mask
            Dt_label = Ds_label
            Dt_Domain_ID = Ds_Domain_ID

            # 模型输出
            # pred_cls, f_rec, f_SR_anchor, f_SR_pairs, pred_dis_invariant = model(Ds_face, Dt_face)
            # pred_cls, pred_cls2, f_SR_anchor, f_SR_pairs, fea_IIE_ori, fea_IIE_rand, pred_dis_invariant, separate_feature, img_separation = model(Ds_face, Dt_face, input_face_next)
            out = model(Ds_face.to("cuda:0"))
            pred_cls = out[0]
            f_SR_anchor = out[1]
            # pred_cls, f_SR_anchor, f_SR_pairs, pred_dis_invariant = model(Ds_face, Dt_face)
            prob_all.extend(pred_cls[:, 1].cpu().data)  # prob[:,1]返回每一行第二列的数，roc_auc_score函数需要最大标签类对应的概率
            label_all.extend(Ds_label.cpu().data)
            f_SR_anchor = f_SR_anchor.reshape(f_SR_anchor.shape[0], -1)
            if batch_index == 0:
                fea_list = f_SR_anchor.cpu().data
                label_list = Ds_label.cpu().data
            else:
                fea_list = torch.cat((fea_list, f_SR_anchor.cpu().data), dim=0)
                label_list = torch.cat((label_list, Ds_label.cpu().data), dim=0)

            """计算分类结果"""
            preds = softmax(pred_cls)
            result_index = getResult(preds, threshold=0.5)  # 根据分类概率计算分类结果
            result_index = result_index.long()
            test_label = Ds_label.long()
            for j in range(len(test_label)):
                if test_label[j] == result_index[j]:
                    confusion_matrix[test_label[j]][test_label[j]] += 1
                else:
                    confusion_matrix[test_label[j]][result_index[j]] += 1

            # 混淆矩阵位置调整
            TP = confusion_matrix[1][1]
            TN = confusion_matrix[0][0]
            FN = confusion_matrix[1][0]
            FP = confusion_matrix[0][1]
            acc_test = float(100.0 * ((TP + TN) / (TP + TN + FN + FP)))
            TPR = 100.0 * (TP / (TP + FN))
            FPR = 100.0 * (FP / (FP + TN))
            TNR = 100 - FPR
            HTER = 100 - (TPR + TNR) / 2

            if ((batch_index + 1) % params.print_freq == 0):
                print('batch[{}/{}]'.format(batch_index + 1, batch_num))
                print('【confusion_matrix】:')
                print(torch.tensor([[TP, FN], [FP, TN]]))
                print('【acc_test:{%f%%}】, 【TPR:{%f%%}】, 【FPR:{%f%%}】, 【TNR:{%f%%}】, 【HTER:{%f%%}】' %
                      (float(acc_test), float(TPR), float(FPR), float(TNR), float(HTER)))

    """绘制T-SNE特征分布可视化"""
    # 定义T-SNE可视化工具。n_components表示降维后嵌入空间的维度，如2或3；init表示嵌入的初始化，可选'pca'或'random'；random_state表示伪随机数发生器种子控制
    ts = manifold.TSNE(n_components=2, init='pca', random_state=0)
    fea_list = fea_list.numpy()
    label_list = label_list.numpy()
    result = ts.fit_transform(fea_list)
    fig = plot_embedding(result, label_list, 't-SNE Embedding')
    number = len(os.listdir("./data/tsne/ori/sbi"))
    np.save(os.path.join("./data/tsne/ori/sbi",str(number) + "_fea.npy"), fea_list)
    np.save(os.path.join("./data/tsne/ori/sbi",str(number) + "_label.npy"), label_list)
    fig.savefig(
        f"./data/tsne/ori/sbi/{number}.png",  # 换png，避免jpg有损压缩
        dpi=300,                     # 高清分辨率，散点更清晰
        bbox_inches='tight',         # 必加：去除多余白边，散点显示在图片中央
        pad_inches=0.1,              # 轻微内边距，避免散点贴边（0也可以）
        facecolor='white',           # 画布背景白（默认，可改black）
        edgecolor='none'             # 无边框
    )
    plt.close(fig)

    # 混淆矩阵位置调整
    TP = confusion_matrix[1][1]
    TN = confusion_matrix[0][0]
    FN = confusion_matrix[1][0]
    FP = confusion_matrix[0][1]
    acc_test = float(100.0 * ((TP + TN) / (TP + TN + FN + FP)))
    TPR = 100.0 * (TP / (TP + FN))
    FPR = 100.0 * (FP / (FP + TN))
    TNR = 100 - FPR
    HTER = 100 - (TPR + TNR) / 2
    print('【confusion_matrix】:')
    print(torch.tensor([[TP, FN], [FP, TN]]))
    print('【ii:{end}】')
    print('【acc_test:{%f%%}】, 【TPR:{%f%%}】, 【FPR:{%f%%}】, 【TNR:{%f%%}】, 【HTER:{%f%%}】' %
          (float(acc_test), float(TPR), float(FPR), float(TNR), float(HTER)))

    """绘制真假样本预测概率的分布图"""
    pred_T = []
    pred_F = []
    for i in range(len(label_all)):
        if label_all[i] == 1:
            pred_T.append(prob_all[i])
        else:
            pred_F.append(prob_all[i])
    bins = np.arange(-0.1, 1.1, 0.01)  # 设置连续的边界值，即直方图的分布区间[0,10],[10,20]...
    # 直方图会进行统计各个区间的数值,normed=True是频率图，默认是频数图,alpha设置透明度，0为完全透明
    frequency_T, _, _ = plt.hist(pred_T, bins, density=True, stacked=True, facecolor='blue', edgecolor='k', alpha=0.5)
    frequency_F, _, _ = plt.hist(pred_F, bins, density=True, stacked=True, facecolor='red', edgecolor='k', alpha=0.5)
    formatter = FuncFormatter(to_percent)
    plt.gca().yaxis.set_major_formatter(formatter)
    fontsize = 15
    plt.xlabel('pred scores', fontsize=fontsize)
    plt.ylabel('probability', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)  # 设置xy轴刻度文字大小
    plt.yticks(fontsize=fontsize)
    plt.xlim([-0.1, 1.1])
    plt.ylim([0, 100])
    plt.legend(["Real", "Fake"], fontsize=fontsize)
    plt.show()

    AUC = roc_auc_score(label_all, prob_all)  # label_all, prob_all形状要一样
    print("AUC:{:.2f}".format(AUC * 100))
    fpr, tpr, thersholds = roc_curve(label_all, prob_all, pos_label=1)
    AUC2 = auc(fpr, tpr)  # 计算AUC
    print("AUC:{:.2f}".format(AUC2 * 100))
    print(len(thersholds))

    # 计算EER和对应的阈值
    diff = []
    for i, value in enumerate(thersholds):
        far = fpr[i]  # 错误接受率
        frr = 1 - tpr[i]  # 错误拒绝率
        diff_temp = abs(far - frr)
        diff.append(diff_temp)
        # print("%f %f %f" % (fpr[i], tpr[i], value))
    min_diff = min(diff)
    min_loc = diff.index(min_diff)
    far_selected = fpr[min_loc] * 100
    frr_selected = (1 - tpr[min_loc]) * 100
    thershold_selected = thersholds[min_loc]
    hter = 0.5 * (far_selected + frr_selected)
    print("EER: far=%f%%, frr=%f%%, thershold_selected=%f" % (far_selected, frr_selected, thershold_selected))
    print("EER=%f%%, HTER=%f%%" % (hter, hter))

    # 绘制ROC曲线
    plt.plot(fpr, tpr, 'k--', label='SPSL (AUC = {0:.2f})'.format(AUC2 * 100), lw=2)
    # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')  # 可以使用中文，但需要导入一些库即字体
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()


def test_model_TSNE_style(model, test_dataloader_list):
    """模型测试"""
    model.eval()

    # 定义统计指标
    confusion_matrix = torch.zeros(3, 3)
    softmax = torch.nn.Softmax(dim=1)  # 计算分类概率，dim=1表示按行计算

    batch_num = int(params.test_num_samples / params.sample_size)
    print('\nbatch_num:{}\n'.format(batch_num))

    # 判断result.txt的目录是否存在，不存在则递归创建目录
    if not os.path.exists(params.result_file_root):
        os.makedirs(params.result_file_root)

    fea_list = []
    label_list = []
    prob_all = []  # 记录预测概率
    label_all = []  # 记录对应的标签
    data_zip = zip(test_dataloader_list[0], test_dataloader_list[1], test_dataloader_list[2])
    with torch.no_grad():
        for batch_index, (D1_T_data, D1_F_data, D1_S_data) in enumerate(tqdm(data_zip)):

            # 修改D1_S_data的label
            label_S_new = 2 * torch.ones_like(D1_S_data[2])
            D1_S_data[2] = label_S_new

            data_list = [D1_T_data, D1_F_data, D1_S_data]
            # 合并数据
            Ds_face, Ds_mask, Ds_label, Ds_Domain_ID, Ds_path, input_face_next, input_mask_next = cat_multiDomains(data_list)
            # Dt_face, Dt_mask, Dt_label, Dt_Domain_ID, Ds_path = cat_multiDomains(data_list)

            # 复制得到目标域Dt
            Dt_face = Ds_face
            Dt_mask = Ds_mask
            Dt_label = Ds_label
            Dt_Domain_ID = Ds_Domain_ID

            # 模型输出
            out = model(Ds_face, Dt_face, input_face_next)
            pred_cls = out[0]
            f_SR_anchor= out[4]
            # pred_cls, f_rec, f_SR_anchor, f_SR_pairs, pred_dis_invariant = model(Ds_face, Dt_face)

            prob_all.extend(pred_cls[:, 1].cpu().data)  # prob[:,1]返回每一行第二列的数，roc_auc_score函数需要最大标签类对应的概率
            label_all.extend(Ds_label.cpu().data)

            if batch_index == 0:
                fea_list = f_SR_anchor.cpu().data
                label_list = Ds_label.cpu().data
            else:
                fea_list = torch.cat((fea_list, f_SR_anchor.cpu().data), dim=0)
                label_list = torch.cat((label_list, Ds_label.cpu().data), dim=0)

            """计算分类结果"""
            preds = softmax(pred_cls)
            result_index = getResult(preds, threshold=0.5)  # 根据分类概率计算分类结果
            result_index = result_index.long()
            test_label = Ds_label.long()
            for j in range(len(test_label)):
                if test_label[j] == result_index[j]:
                    confusion_matrix[test_label[j]][test_label[j]] += 1
                else:
                    confusion_matrix[test_label[j]][result_index[j]] += 1

            # 混淆矩阵位置调整
            TP = confusion_matrix[1][1]
            TN = confusion_matrix[0][0]
            FN = confusion_matrix[1][0]
            FP = confusion_matrix[0][1]
            acc_test = float(100.0 * ((TP + TN) / (TP + TN + FN + FP)))
            TPR = 100.0 * (TP / (TP + FN))
            FPR = 100.0 * (FP / (FP + TN))
            TNR = 100 - FPR
            HTER = 100 - (TPR + TNR) / 2

            if ((batch_index + 1) % params.print_freq == 0):
                print('batch[{}/{}]'.format(batch_index + 1, batch_num))
                print('【confusion_matrix】:')
                print(torch.tensor([[TP, FN], [FP, TN]]))
                print('【acc_test:{%f%%}】, 【TPR:{%f%%}】, 【FPR:{%f%%}】, 【TNR:{%f%%}】, 【HTER:{%f%%}】' %
                      (float(acc_test), float(TPR), float(FPR), float(TNR), float(HTER)))

    """绘制T-SNE特征分布可视化"""
    # 定义T-SNE可视化工具。n_components表示降维后嵌入空间的维度，如2或3；init表示嵌入的初始化，可选'pca'或'random'；random_state表示伪随机数发生器种子控制
    ts = manifold.TSNE(n_components=2, init='pca', random_state=0)
    result = ts.fit_transform(fea_list)
    fig = plot_embedding(result, label_list, 't-SNE Embedding')
    plt.show()

    # 混淆矩阵位置调整
    TP = confusion_matrix[1][1]
    TN = confusion_matrix[0][0]
    FN = confusion_matrix[1][0]
    FP = confusion_matrix[0][1]
    acc_test = float(100.0 * ((TP + TN) / (TP + TN + FN + FP)))
    TPR = 100.0 * (TP / (TP + FN))
    FPR = 100.0 * (FP / (FP + TN))
    TNR = 100 - FPR
    HTER = 100 - (TPR + TNR) / 2
    print('【confusion_matrix】:')
    print(torch.tensor([[TP, FN], [FP, TN]]))
    print('【ii:{end}】')
    print('【acc_test:{%f%%}】, 【TPR:{%f%%}】, 【FPR:{%f%%}】, 【TNR:{%f%%}】, 【HTER:{%f%%}】' %
          (float(acc_test), float(TPR), float(FPR), float(TNR), float(HTER)))

    """绘制真假样本预测概率的分布图"""
    pred_T = []
    pred_F = []
    for i in range(len(label_all)):
        if label_all[i] == 1:
            pred_T.append(prob_all[i])
        else:
            pred_F.append(prob_all[i])
    bins = np.arange(-0.1, 1.1, 0.01)  # 设置连续的边界值，即直方图的分布区间[0,10],[10,20]...
    # 直方图会进行统计各个区间的数值,normed=True是频率图，默认是频数图,alpha设置透明度，0为完全透明
    frequency_T, _, _ = plt.hist(pred_T, bins, density=True, stacked=True, facecolor='blue', edgecolor='k', alpha=0.5)
    frequency_F, _, _ = plt.hist(pred_F, bins, density=True, stacked=True, facecolor='red', edgecolor='k', alpha=0.5)
    formatter = FuncFormatter(to_percent)
    plt.gca().yaxis.set_major_formatter(formatter)
    fontsize = 15
    plt.xlabel('pred scores', fontsize=fontsize)
    plt.ylabel('probability', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)  # 设置xy轴刻度文字大小
    plt.yticks(fontsize=fontsize)
    plt.xlim([-0.1, 1.1])
    plt.ylim([0, 100])
    plt.legend(["Real", "Fake"], fontsize=fontsize)
    plt.show()

    AUC = roc_auc_score(label_all, prob_all)  # label_all, prob_all形状要一样
    print("AUC:{:.2f}".format(AUC * 100))
    fpr, tpr, thersholds = roc_curve(label_all, prob_all, pos_label=1)
    AUC2 = auc(fpr, tpr)  # 计算AUC
    print("AUC:{:.2f}".format(AUC2 * 100))
    print(len(thersholds))

    # 计算EER和对应的阈值
    diff = []
    for i, value in enumerate(thersholds):
        far = fpr[i]  # 错误接受率
        frr = 1 - tpr[i]  # 错误拒绝率
        diff_temp = abs(far - frr)
        diff.append(diff_temp)
        # print("%f %f %f" % (fpr[i], tpr[i], value))
    min_diff = min(diff)
    min_loc = diff.index(min_diff)
    far_selected = fpr[min_loc] * 100
    frr_selected = (1 - tpr[min_loc]) * 100
    thershold_selected = thersholds[min_loc]
    hter = 0.5 * (far_selected + frr_selected)
    print("EER: far=%f%%, frr=%f%%, thershold_selected=%f" % (far_selected, frr_selected, thershold_selected))
    print("EER=%f%%, HTER=%f%%" % (hter, hter))

    # 绘制ROC曲线
    plt.plot(fpr, tpr, 'k--', label='SPSL (AUC = {0:.2f})'.format(AUC2 * 100), lw=2)
    # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')  # 可以使用中文，但需要导入一些库即字体
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()


def test_model_TSNE_multiDomains(model, test_dataloader_list):
    """模型测试，同时绘制多个测试集"""
    model.eval()

    # 定义统计指标
    confusion_matrix = torch.zeros(2, 2)
    # confusion_matrix = torch.zeros(10, 10)      # 多个测试域，赋值不同label
    softmax = torch.nn.Softmax(dim=1)  # 计算分类概率，dim=1表示按行计算

    batch_num = int(params.test_num_samples / params.sample_size)
    print('\nbatch_num:{}\n'.format(batch_num))

    # 判断result.txt的目录是否存在，不存在则递归创建目录
    if not os.path.exists(params.result_file_root):
        os.makedirs(params.result_file_root)

    fea_list = []
    label_list = []
    prob_all = []  # 记录预测概率
    label_all = []  # 记录对应的标签

    data_zip = zip(test_dataloader_list[0], test_dataloader_list[1], test_dataloader_list[2], test_dataloader_list[3],
                   test_dataloader_list[4], test_dataloader_list[5], test_dataloader_list[6], test_dataloader_list[7],
                   test_dataloader_list[8], test_dataloader_list[9])

    with torch.no_grad():
        for batch_index, (
                D1_T_data, D1_F_data, D2_T_data, D2_F_data, D3_T_data, D3_F_data, D4_T_data, D4_F_data, D5_T_data,
                D5_F_data) in enumerate(tqdm(data_zip)):

            # 修改data的label
            D2_F_data[2] = 2 * torch.ones_like(D2_F_data[2])
            D2_T_data[2] = 3 * torch.ones_like(D2_T_data[2])
            D3_F_data[2] = 4 * torch.ones_like(D3_F_data[2])
            D3_T_data[2] = 5 * torch.ones_like(D3_T_data[2])
            D4_F_data[2] = 6 * torch.ones_like(D4_F_data[2])
            D4_T_data[2] = 7 * torch.ones_like(D4_T_data[2])
            D5_F_data[2] = 8 * torch.ones_like(D5_F_data[2])
            D5_T_data[2] = 9 * torch.ones_like(D5_T_data[2])

            data_list = [D1_T_data, D1_F_data, D2_T_data, D2_F_data, D3_T_data, D3_F_data, D4_T_data, D4_F_data,
                         D5_T_data, D5_F_data]

            # 合并数据
            Ds_face, Ds_mask, Ds_label, Ds_Domain_ID, Ds_path, input_face_next, input_mask_next = cat_multiDomains(data_list)
            # Dt_face, Dt_mask, Dt_label, Dt_Domain_ID, Ds_path = cat_multiDomains(data_list)

            # 复制得到目标域Dt
            Dt_face = Ds_face
            Dt_mask = Ds_mask
            Dt_label = Ds_label
            Dt_Domain_ID = Ds_Domain_ID

            # 模型输出
            pred_cls, pred_cls2, f_SR_anchor, f_SR_pairs, fea_IIE_ori, fea_IIE_rand, pred_dis_invariant, separate_feature, img_separation = model(
                Ds_face, Dt_face)
            # pred_cls, f_rec, f_SR_anchor, f_SR_pairs, pred_dis_invariant = model(Ds_face, Dt_face)

            prob_all.extend(pred_cls[:, 1].cpu().data)  # prob[:,1]返回每一行第二列的数，roc_auc_score函数需要最大标签类对应的概率
            label_all.extend(Ds_label.cpu().data)

            if batch_index == 0:
                fea_list = f_SR_anchor.cpu().data
                label_list = Ds_label.cpu().data
            else:
                fea_list = torch.cat((fea_list, f_SR_anchor.cpu().data), dim=0)
                label_list = torch.cat((label_list, Ds_label.cpu().data), dim=0)

            if ((batch_index + 1) % params.print_freq == 0):
                print('batch[{}/{}]'.format(batch_index + 1, batch_num))

    """绘制T-SNE特征分布可视化"""
    # 定义T-SNE可视化工具。n_components表示降维后嵌入空间的维度，如2或3；init表示嵌入的初始化，可选'pca'或'random'；random_state表示伪随机数发生器种子控制
    ts = manifold.TSNE(n_components=2, init='pca', random_state=0)
    result = ts.fit_transform(fea_list)
    # fig = plot_embedding3(result, label_list, 't-SNE Embedding')
    # fig = plot_embedding3_v0(result, label_list, 't-SNE Embedding')
    fig = plot_embedding3_v2_new(result, label_list, 't-SNE Embedding')  # 输出FF++和DFDC两个库（包括SFFS）
    plt.show()

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ------------------------------------------------------------
# 2D Discrete Cosine Transform (DCT) and Inverse DCT (IDCT)
# 使用可分离的 DCT 矩阵实现，支持梯度传播
# ------------------------------------------------------------
class DCT2D(nn.Module):
    def __init__(self, height, width, norm='ortho'):
        super().__init__()
        self.h, self.w = height, width
        self.norm = norm
        # 预计算 DCT 矩阵
        self.register_buffer('dct_h', self._dct_matrix(height))
        self.register_buffer('dct_w', self._dct_matrix(width))

    def _dct_matrix(self, n):
        """生成 n x n 的 DCT 矩阵 (正交归一化)"""
        x, y = torch.meshgrid(torch.arange(n), torch.arange(n), indexing='ij')
        matrix = torch.cos(math.pi / n * (x + 0.5) * y)
        matrix[:, 0] = matrix[:, 0] / math.sqrt(2)   # 调整 DC 分量
        if self.norm == 'ortho':
            matrix *= math.sqrt(2.0 / n)
        else:
            matrix *= 1.0
        return matrix.float()

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        assert H == self.h and W == self.w
        # 对行做 DCT: (B, C, H, W) -> (B, C, H, W)
        x = torch.matmul(self.dct_h, x.permute(0,1,3,2)).permute(0,1,3,2)
        # 对列做 DCT
        x = torch.matmul(self.dct_w, x)
        return x

class IDCT2D(nn.Module):
    def __init__(self, height, width, norm='ortho'):
        super().__init__()
        self.h, self.w = height, width
        self.norm = norm
        # 预计算 IDCT 矩阵 (DCT 矩阵的转置)
        self.register_buffer('idct_h', self._dct_matrix(height).t())
        self.register_buffer('idct_w', self._dct_matrix(width).t())

    def _dct_matrix(self, n):
        """与 DCT 矩阵相同 (用于生成转置)"""
        x, y = torch.meshgrid(torch.arange(n), torch.arange(n), indexing='ij')
        matrix = torch.cos(math.pi / n * (x + 0.5) * y)
        matrix[:, 0] = matrix[:, 0] / math.sqrt(2)
        if self.norm == 'ortho':
            matrix *= math.sqrt(2.0 / n)
        else:
            matrix *= 1.0
        return matrix.float()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.h and W == self.w
        # 对行做 IDCT
        x = torch.matmul(self.idct_h, x.permute(0,1,3,2)).permute(0,1,3,2)
        # 对列做 IDCT
        x = torch.matmul(self.idct_w, x)
        return x

# ------------------------------------------------------------
# IIEM: Inter-frame Illumination Exchange Module
# 使用可学习频域滤波器分离光照/反射，并交换低频分量
# ------------------------------------------------------------
class IIEM(nn.Module):
    def __init__(self, channels, height, width):
        super().__init__()
        self.channels = channels
        self.h, self.w = height, width
        # 固定基滤波器 (低通和高通)
        # 简单起见，低通使用高斯掩模，高通为 1 - 低通
        self.register_buffer('base_low', self._gaussian_mask(height, width, sigma=0.2))
        self.register_buffer('base_high', 1 - self.base_low)

        # 可学习部分 (初始化为0，使初始滤波器接近固定基)
        self.learned_low = nn.Parameter(torch.zeros(1, 1, height, width))
        self.learned_high = nn.Parameter(torch.zeros(1, 1, height, width))

        # DCT 和 IDCT 模块
        self.dct = DCT2D(height, width)
        self.idct = IDCT2D(height, width)

        # 预训练标志 (简单起见，我们在此不实现预训练逻辑，实际使用时需加载预训练权重)
        self.frozen = False

    def _gaussian_mask(self, h, w, sigma):
        """生成二维高斯低通掩模 (频域)"""
        y, x = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing='ij')
        d = torch.sqrt(x*x + y*y)
        mask = torch.exp(- (d*d) / (2*sigma*sigma))
        return mask.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

    def forward(self, f_i, f_j):
        """
        f_i, f_j: 两个帧的特征 (B, C, H, W)
        返回重组后的特征 f_rec_i, f_rec_j
        """
        # 转换到 DCT 域
        F_i = self.dct(f_i)   # (B, C, H, W)
        F_j = self.dct(f_j)

        # 构建完整滤波器: Φ = base + tanh(learned)   (使值在 [-1,1] 附近)
        phi_low = self.base_low + torch.tanh(self.learned_low)
        phi_high = self.base_high + torch.tanh(self.learned_high)

        # 分离低频和高频
        F_i_low = F_i * phi_low
        F_i_high = F_i * phi_high
        F_j_low = F_j * phi_low
        F_j_high = F_j * phi_high

        # 交换低频部分
        F_rec_i = F_j_low + F_i_high
        F_rec_j = F_i_low + F_j_high

        # 转换回空间域
        f_rec_i = self.idct(F_rec_i)
        f_rec_j = self.idct(F_rec_j)
        return f_rec_i, f_rec_j

    def freeze(self):
        """预训练后冻结参数"""
        for p in self.parameters():
            p.requires_grad = False
        self.frozen = True

# ------------------------------------------------------------
# IIFM: Intra-frame Illumination Fusion Module
# 交叉模态注意力融合 RGB 和 LPLGF 特征
# ------------------------------------------------------------
class IIFM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.reduction = reduction
        # 两个输入分支的通道数均为 in_channels
        # key 分支：1x1 降维 + 共享 3x3 卷积
        self.key_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.Conv2d(in_channels // reduction, in_channels // reduction, 3, padding=1, bias=False)
        )
        # value 分支：1x1 卷积保持通道数
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1, bias=False)

        # 可学习的权重矩阵 W_A, W_B (在注意力中)
        # 注意：这里我们用 1x1 卷积来模拟矩阵乘法，因为 N=H*W 未知
        # 简化：使用两个全连接层，但为了适应任意尺寸，使用 1x1 conv 再 reshape
        self.W_A = nn.Conv1d(1, 1, kernel_size=1, bias=False)  # 实际上用于缩放，我们采用更直接的方式
        self.W_B = nn.Conv1d(1, 1, kernel_size=1, bias=False)
        # 更简单的做法：直接使用可学习标量，但论文中是矩阵
        # 为简化，我们假设注意力图计算后不再乘以额外的 W，或者使用 1x1 conv
        # 实际上，可以通过一个可学习的权重张量，但我们这里使用一个小的卷积来模拟
        self.attn_conv = nn.Conv2d(in_channels // reduction, in_channels // reduction, 1)

    def forward(self, f_A, f_B):
        """
        f_A, f_B: (B, C, H, W) 来自 RGB 和 LPLGF 分支
        返回融合后的 f_A', f_B'
        """
        B, C, H, W = f_A.shape
        # 生成 key 和 value
        key_A = self.key_conv(f_A)          # (B, C/r, H, W)
        key_B = self.key_conv(f_B)
        val_A = self.value_conv(f_A)        # (B, C, H, W)
        val_B = self.value_conv(f_B)

        # 展平空间维度
        kA = key_A.view(B, C//self.reduction, -1)   # (B, C/r, N)
        kB = key_B.view(B, C//self.reduction, -1)
        vA = val_A.view(B, C, -1)                   # (B, C, N)
        vB = val_B.view(B, C, -1)
        N = kA.shape[-1]

        # 计算相关矩阵 C = kA^T @ kB  (B, N, N)
        C = torch.matmul(kA.transpose(1,2), kB)

        # 生成注意力图 A_A, A_B (B, N, N)
        # 简化：不引入额外可学习矩阵 W，因为可以用 conv 但维度不匹配
        # 我们直接用 C 经过 softmax，因为 W 可以融合进 C 的学习中
        A_A = F.softmax(C, dim=-1)
        A_B = F.softmax(C.transpose(1,2), dim=-1)

        # 加权求和: R_A = vB @ A_A, R_B = vA @ A_B
        R_A = torch.matmul(vB, A_A)          # (B, C, N)
        R_B = torch.matmul(vA, A_B)

        # 恢复空间维度
        R_A = R_A.view(B, C, H, W)
        R_B = R_B.view(B, C, H, W)

        # 残差连接
        f_A_out = f_A + R_A
        f_B_out = f_B + R_B
        return f_A_out, f_B_out

# ------------------------------------------------------------
# IIDM: Inter-frame Illumination Discrepancy Module
# 通过频域分解、交换和注意力增强帧间差异
# ------------------------------------------------------------
class IIDM(nn.Module):
    def __init__(self, channels, height, width):
        super().__init__()
        self.channels = channels
        self.h, self.w = height, width

        # 固定基滤波器 (与 IIEM 类似，但可以独立)
        self.register_buffer('base_low', self._gaussian_mask(height, width, sigma=0.2))
        self.register_buffer('base_high', 1 - self.base_low)
        self.learned_low = nn.Parameter(torch.zeros(1, 1, height, width))
        self.learned_high = nn.Parameter(torch.zeros(1, 1, height, width))

        self.dct = DCT2D(height, width)
        self.idct = IDCT2D(height, width)

        # 通道注意力
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 16, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 16, channels, 1),
            nn.Sigmoid()
        )
        # 空间注意力
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def _gaussian_mask(self, h, w, sigma):
        y, x = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing='ij')
        d = torch.sqrt(x*x + y*y)
        mask = torch.exp(- (d*d) / (2*sigma*sigma))
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, f_i, f_j, f_rec_i, f_rec_j):
        """
        f_i, f_j: 原始帧特征 (B, C, H, W)
        f_rec_i, f_rec_j: 由 IIEM 产生的重组特征
        返回增强后的特征 f_out_i, f_out_j (用于后续分类)
        """
        # ---- 频域分解与交换 (与 IIEM 类似，但使用独立滤波器) ----
        phi_low = self.base_low + torch.tanh(self.learned_low)
        phi_high = self.base_high + torch.tanh(self.learned_high)

        F_i = self.dct(f_i)
        F_j = self.dct(f_j)
        # 交换低频
        F_i_swapped = F_i * phi_high + F_j * phi_low   # 注意：这里与 IIEM 公式一致，但论文中是 f_i 保留高频，加上 f_j 的低频
        F_j_swapped = F_j * phi_high + F_i * phi_low
        f_i_swapped = self.idct(F_i_swapped)
        f_j_swapped = self.idct(F_j_swapped)

        # ---- 计算残差 ----
        delta_i = f_i - f_i_swapped
        delta_j = f_j - f_j_swapped

        # ---- 通道和空间注意力增强 ----
        # 对残差应用通道注意力
        ca_i = self.channel_att(delta_i)
        ca_j = self.channel_att(delta_j)
        # 对残差应用空间注意力 (需要构建空间特征)
        # 空间注意力输入: 平均池化和最大池化沿通道维
        avg_i = torch.mean(delta_i, dim=1, keepdim=True)
        max_i, _ = torch.max(delta_i, dim=1, keepdim=True)
        spatial_input_i = torch.cat([avg_i, max_i], dim=1)
        sa_i = self.spatial_att(spatial_input_i)

        avg_j = torch.mean(delta_j, dim=1, keepdim=True)
        max_j, _ = torch.max(delta_j, dim=1, keepdim=True)
        spatial_input_j = torch.cat([avg_j, max_j], dim=1)
        sa_j = self.spatial_att(spatial_input_j)

        # 结合注意力和原始重组特征
        f_out_i = f_rec_i * ca_i * sa_i + f_rec_i   # 残差连接
        f_out_j = f_rec_j * ca_j * sa_j + f_rec_j

        return f_out_i, f_out_j


if __name__ == "__main__":
    B, C, H, W = 2, 64, 16, 16   # 假设特征尺寸
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建随机输入
    f_i = torch.randn(B, C, H, W).to(device)
    f_j = torch.randn(B, C, H, W).to(device)
    f_A = torch.randn(B, C, H, W).to(device)   # RGB 特征
    f_B = torch.randn(B, C, H, W).to(device)   # LPLGF 特征

    # 初始化模块
    iiem = IIEM(C, H, W).to(device)
    iifm = IIFM(C, reduction=16).to(device)
    iidm = IIDM(C, H, W).to(device)

    # IIEM 前向
    f_rec_i, f_rec_j = iiem(f_i, f_j)
    print("IIEM output shapes:", f_rec_i.shape, f_rec_j.shape)

    # IIFM 前向
    f_A_out, f_B_out = iifm(f_A, f_B)
    print("IIFM output shapes:", f_A_out.shape, f_B_out.shape)

    # IIDM 前向
    f_out_i, f_out_j = iidm(f_i, f_j, f_rec_i, f_rec_j)
    print("IIDM output shapes:", f_out_i.shape, f_out_j.shape)