from __future__ import print_function, division, absolute_import
import math
import torch
import torch.nn as nn
import albumentations as alb
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms as T
from torchvision import transforms

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import random
from glob import glob
import json

import albumentations as A
from torch.utils import data

from utils.utils import norm
import os

import src.params as params
import time

import warnings
import sys

warnings.filterwarnings("ignore")



"""训练Xception的数据集，以RGB图像为输入"""

class RGB_Dataset(data.Dataset):
    """
    DeepfakeDataset_RGB的数据类，使用Input_txt文件夹的数据路径格式吗，真假人脸数据在同一个文件内
    每一行数据格式为: [img_path, mask_path, label, [ymin, ymax, xmin, xmax], [x0,y0, x1,y1, ... , x68,y68]]
    """

    def __init__(self, dataPath=None, transform=True, sampleType='all', faceRotate=False, mode_aug=False,
                 image_size=256):
        """获取正负样本图像的路径"""
        super(RGB_Dataset, self).__init__()  # 继承父类的初始化操作

        imgs_path = []
        imgs_path_A = []
        imgs_path_T = []
        imgs_path_F = []

        if dataPath is not None:
            f_true = open(dataPath)
            for line in f_true:
                pathData = line.rstrip()  # rstrip()用来删除字符串末尾的指定字符（默认为空格）
                words = pathData.split(',')  # 分割路径，提取path，label，rect等信息

                """根据label将正负样本分开"""
                if int(words[2]) == 0:
                    imgs_path_F.append(
                        (words[0], int(words[2]), [int(words[3]), int(words[4]), int(words[5]), int(words[6])],
                         words[7:7 + 68 * 2]))
                else:
                    imgs_path_T.append(
                        (words[0], int(words[2]), [int(words[3]), int(words[4]), int(words[5]), int(words[6])],
                         words[7:7 + 68 * 2]))
                imgs_path_A.append(
                    (words[0], int(words[2]), [int(words[3]), int(words[4]), int(words[5]), int(words[6])],
                     words[7:7 + 68 * 2]))

        if sampleType == 'all':
            imgs_path = imgs_path_A
        if sampleType == 'real':
            imgs_path = imgs_path_T
        if sampleType == 'fake':
            imgs_path = imgs_path_F

        self.imgs_path = imgs_path
        self.transform = transform
        self.mode_aug = mode_aug
        self.aug_transforms = A.Compose([
            A.Compose([
                A.RGBShift((-20, 20), (-20, 20), (-20, 20), p=0.3),
                A.HueSaturationValue(hue_shift_limit=(-0.3, 0.3), sat_shift_limit=(-0.3, 0.3),
                                     val_shift_limit=(-0.3, 0.3), p=0.3),
                A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.3),
            ], p=1),
            A.ImageCompression(quality_lower=40, quality_upper=100, p=0.3),  # 图像压缩因子
            A.OneOf([
                A.MotionBlur(p=0.3),  # 使用随机大小的内核将运动模糊应用于输入图像。
                A.MedianBlur(blur_limit=3, p=0.3),  # 中值滤波
                A.Blur(blur_limit=3, p=0.3),  # 使用随机大小的内核模糊输入图像。
            ], p=0.5),
        ], p=1)
        self.transform_Spatial = T.Compose([
            T.Resize([image_size, image_size]),
            T.ToTensor(),  # change image to tensor
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # Normalize, x=(x-mean)/std
            # T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize, x=(x-mean)/std
        ])
        self.image_size = image_size
        self.faceRotate = faceRotate

    def __getitem__(self, index):
        """
        返回一张图片数据
        :param index: 图片序号
        :return: PIL图片数据
        """
        # 这里的rect格式是[ymin, ymax, xmin, xmax]
        path, label, rect, landmarks = self.imgs_path[index]

        img = cv2.imread(path)  # cv格式的BGR图像数据（适合读取图像后需要进行一系列图像处理的方法，因为需要用的OpenCV）

        # 根据68个特征点对人脸进行对齐矫正
        landmarks_ori = []
        for i in range(len(landmarks)):
            if i % 2 == 0:
                landmarks_ori.append([int(landmarks[i]), int(landmarks[i + 1])])
        # 是否进行人脸矫正
        if self.faceRotate == True:
            image_rotate, landmarks_rotate = self.face_rotate(img, landmarks_ori)
        else:
            image_rotate, landmarks_rotate = img, np.array(landmarks_ori)
        image_rotate_shape = image_rotate.shape
        rows = image_rotate_shape[0]  # height(rows) of image
        cols = image_rotate_shape[1]  # width(colums) of image

        # 根据矫正后的人脸特征点选取人脸框位置
        point_x = []
        point_y = []
        for i in range(len(landmarks_rotate)):
            point_x.append(int(landmarks_rotate[i][0]))
            point_y.append(int(landmarks_rotate[i][1]))
        ymin = min(point_y)
        ymax = max(point_y)
        xmin = min(point_x)
        xmax = max(point_x)
        ymin = max(ymin, 0)
        ymax = min(ymax, rows)
        xmin = max(xmin, 0)
        xmax = min(xmax, cols)

        # 将人脸向外扩大k倍
        ymin2, ymax2, xmin2, xmax2 = self.face_resize(rows, cols, ymin, ymax, xmin, xmax, k=1.3)
        img_face = image_rotate[ymin2:ymax2, xmin2:xmax2]  # cv的裁剪图像方式，裁剪坐标为[ymin:ymax, xmin:xmax]

        # 将PIL格式的图像数据转化为numpy格式，因为OpenCV计算DCT图的函数输入是numpy float32格式
        img_face = np.asarray(img_face)

        # 数据增强
        if self.mode_aug == True:
            img_face_aug = self.aug_transforms(image=img_face)
            img_face_aug = img_face_aug["image"]  # 增强后的数据拥有多个属性，需要取出"image"
        else:
            img_face_aug = img_face
        # 图像尺寸和模式调整
        BGR_img_face = cv2.resize(img_face_aug, (self.image_size, self.image_size))
        RGB_img_face = cv2.cvtColor(BGR_img_face, cv2.COLOR_BGR2RGB)

        # 将得到的numpy格式的图转换为PIL图像格式，因为transform的输入是PIL格式的图像
        img_face_input = Image.fromarray(np.uint8(RGB_img_face))

        if self.transform:
            img_face_input = self.transform_Spatial(img_face_input)
        return img_face_input, label

    def __len__(self):
        """
        返回数据集中图像的数量
        :return:
        """
        return len(self.imgs_path)

    def face_resize(self, rows, cols, ymin, ymax, xmin, xmax, k=2):
        """获得k倍人脸"""
        height = ymax - ymin
        width = xmax - xmin
        height1 = round(k * height)
        width1 = round(k * width)
        center_y = round((ymin + ymax) / 2)
        center_x = round((xmin + xmax) / 2)
        ymin = round(center_y - height1 / 2)
        ymax = round(center_y + height1 / 2)
        xmin = round(center_x - width1 / 2)
        xmax = round(center_x + width1 / 2)

        ymin = max(ymin, 0)
        ymax = min(ymax, rows)
        xmin = max(xmin, 0)
        xmax = min(xmax, cols)

        return ymin, ymax, xmin, xmax

    def face_rotate(self, image, landmarks):
        """人脸对齐矫正"""
        landmarks = np.array(landmarks)
        # rotation angle计算旋转角度
        left_eye_corner = landmarks[36]  # 左眼外角
        right_eye_corner = landmarks[45]  # 右眼外角
        # 计算偏转角
        radian = np.arctan((left_eye_corner[1] - right_eye_corner[1]) / (left_eye_corner[0] - right_eye_corner[0]))
        # print('radian：' + str(radian))
        degree = math.degrees(radian)
        # print('degree：'+str(degree))

        if abs(degree) > 20:
            return image, landmarks
        else:
            # image size after rotating
            height, width, _ = image.shape
            cos = math.cos(radian)
            sin = math.sin(radian)

            new_w = width * abs(cos) + height * abs(sin)
            new_h = width * abs(sin) + height * abs(cos)
            if np.isnan(new_w):
                return image, landmarks
            elif np.isnan(new_h):
                return image, landmarks
            else:
                new_w = int(new_w)
                new_h = int(new_h)

                # translation
                Tx = new_w // 2 - width // 2
                Ty = new_h // 2 - height // 2

                # affine matrix
                M = np.array([[cos, sin, (1 - cos) * width / 2. - sin * height / 2. + Tx],
                              [-sin, cos, sin * width / 2. + (1 - cos) * height / 2. + Ty]])

                # 用白色填充仿射变换后的边界
                image_rotate = cv2.warpAffine(image, M, (new_w, new_h), borderValue=(255, 255, 255))

                landmarks = np.concatenate([landmarks, np.ones((landmarks.shape[0], 1))], axis=1)
                landmarks_rotate = np.dot(M, landmarks.T).T
                return image_rotate, landmarks_rotate


class RGB_Face_Dataset(data.Dataset):
    """用于训练ISDN网络的数据集，返回RGB人脸图像、label、mask和数据域的ID.——ljc,2023.4.4"""

    def __init__(self, dataPath=None, transform=True, sampleType='all', maskType='rect',
                 faceRotate=False, mode_aug=False, image_size=256, mask_size=256, domain_ID=0, model_sbi=False, sbi_aug=False, spsl=False):
        """
        用于训练ISDN网络的数据集，返回RGB人脸图像、label和对应的mask
        :param dataPath:数据集的txt文件
        :param transform:是否在数据集内使用torch变换
        :param sampleType:返回样本的类型，全部样本、正样本或负样本
        :param maskType:掩膜mask的形状，矩形或者不规则人脸形状
        :param mode_aug:是否使用数据增强
        :param image_size:图像尺寸
        :param domain_ID:数据域的ID
        """
        super(RGB_Face_Dataset, self).__init__()  # 继承父类的初始化操作

        # 用于保存每一行的样本路径，路径格式为”path,label,rect(i)“
        imgs_path = []
        imgs_path_A = []
        imgs_path_T = []
        imgs_path_F = []
        imgs_path_A_next = []
        imgs_path_T_next = []
        imgs_path_F_next = []
        imgs_path_next = []

        if dataPath is not None:
            lines = open(dataPath).readlines()
            total_lines = len(lines)
            next_number = 1
            for idx in range(total_lines - next_number):  # 确保 idx + interval 不越界
                line = lines[idx].rstrip()
                next_line = lines[idx + next_number].rstrip()

                words = line.split(',')
                next_words = next_line.split(',')

                folder_path = os.path.dirname(words[0])
                next_folder_path = os.path.dirname(next_words[0])

                # 保持原逻辑：只有同一文件夹内的样本才配对
                if next_folder_path == folder_path:
                    """根据label将正负样本分开"""
                    if int(words[2]) == 0:
                        imgs_path_F.append(
                            (words[0], int(words[2]), [int(words[3]), int(words[4]), int(words[5]), int(words[6])],
                             words[7:7 + 68 * 2]))
                    else:
                        imgs_path_T.append(
                            (words[0], int(words[2]), [int(words[3]), int(words[4]), int(words[5]), int(words[6])],
                             words[7:7 + 68 * 2]))
                    imgs_path_A.append(
                        (words[0], int(words[2]), [int(words[3]), int(words[4]), int(words[5]), int(words[6])],
                         words[7:7 + 68 * 2]))
                    if int(next_words[2]) == 0:
                        imgs_path_F_next.append((next_words[0], int(next_words[2]),
                                                 [int(next_words[3]), int(next_words[4]), int(next_words[5]),
                                                  int(next_words[6])], next_words[7:7 + 68 * 2]))
                    else:
                        imgs_path_T_next.append((next_words[0], int(next_words[2]),
                                                 [int(next_words[3]), int(next_words[4]), int(next_words[5]),
                                                  int(next_words[6])], next_words[7:7 + 68 * 2]))
                    imgs_path_A_next.append((next_words[0], int(next_words[2]),
                                             [int(next_words[3]), int(next_words[4]), int(next_words[5]),
                                              int(next_words[6])], next_words[7:7 + 68 * 2]))

        if sampleType == 'all':
            imgs_path = imgs_path_A
            imgs_path_next = imgs_path_A_next
        elif sampleType == 'real':
            imgs_path = imgs_path_T
            imgs_path_next = imgs_path_T_next
        elif sampleType == 'fake':
            imgs_path = imgs_path_F
            imgs_path_next = imgs_path_F_next

        self.imgs_path = imgs_path
        self.imgs_path_next = imgs_path_next
        self.transform = transform
        self.spsl = spsl
        self.aug_transforms = A.ReplayCompose([
            A.Compose([
                A.RGBShift(r_shift_limit=(-20, 20), g_shift_limit=(-20, 20), b_shift_limit=(-20, 20), p=0.3),
                A.HueSaturationValue(hue_shift_limit=(-0.3, 0.3), sat_shift_limit=(-0.3, 0.3),
                                     val_shift_limit=(-0.3, 0.3), p=0.3),
                A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.3),
            ], p=1),
            A.ImageCompression(quality_lower=40, quality_upper=100, p=0.3),
            A.OneOf([
                A.MotionBlur(p=0.3),
                A.MedianBlur(blur_limit=3, p=0.3),
                A.Blur(blur_limit=3, p=0.3),
            ], p=0.5),
        ], p=1)

        self.transform_Spatial = T.Compose([
            T.Resize([image_size, image_size]),
            T.ToTensor(),  # change image to tensor
            # T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # Normalize, x=(x-mean)/std
            # T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize, x=(x-mean)/std
        ])
        self.transform_mask = T.Compose([
            T.Resize([image_size, image_size]),
            T.ToTensor(),  # change image to tensor
        ])
        self.mode_aug = mode_aug
        self.maskType = maskType
        self.faceRotate = faceRotate
        self.image_size = image_size
        self.mask_size = mask_size
        self.sbi_aug = sbi_aug
        self.model_sbi = model_sbi
        self.domain_ID = domain_ID
        self.source_transforms = A.ReplayCompose([
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

    def randaffine(self, img, mask):
        f = alb.Affine(
            translate_percent={'x': (-0.03, 0.03), 'y': (-0.015, 0.015)},
            scale=[0.95, 1 / 0.95],
            fit_output=False,
            p=1)

        g = alb.ElasticTransform(
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

    def self_blending(self, img, landmark, replay_data=None):
        # 生成原始 mask
        mask = np.zeros_like(img[:, :, 0]).astype(np.float32)
        cv2.fillConvexPoly(mask, cv2.convexHull(landmark.astype(np.int32)), 1.)
        
        source = img.copy()
        
        if replay_data is None:
            # 第一张图：正常增强并记录参数
            if np.random.rand() < 0.5:
                # 记录 source 的增强
                res = self.source_transforms(image=source.astype(np.uint8))
                source = res['image']
                applied_replay = res['replay'] # 获取本次增强的参数
                aug_target = 'source'
            else:
                # 记录 img 的增强
                res = self.source_transforms(image=img.astype(np.uint8))
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
        source, mask = self.randaffine(source, mask)
        mask = np.expand_dims(mask, axis=-1)
        img_blended = (source * mask + img * (1 - mask)).astype(np.uint8)

        # 返回结果时带上增强参数，供下一张图使用
        return img.astype(np.uint8), img_blended.astype(np.uint8), mask.squeeze(), {"replay": applied_replay, "target": aug_target}

    def __getitem__(self, index):
        """
        返回一张图片数据
        :param index: 图片序号
        :return: PIL图片数据
        """
        # 这里的rect格式是[ymin, ymax, xmin, xmax]
        path, label, rect, landmarks = self.imgs_path[index]

        path_next, label_next, rect_next, landmarks_next = self.imgs_path_next[index]
        img = cv2.imread(path)  # cv格式的BGR图像数据（适合读取图像后需要进行一系列图像处理的方法，因为需要用的OpenCV）
        img_next = cv2.imread(path_next)

        # 根据68个特征点对人脸进行对齐矫正
        landmarks_ori = []
        for i in range(len(landmarks)):
            if i % 2 == 0:
                landmarks_ori.append([int(landmarks[i]), int(landmarks[i + 1])])

        landmarks_next_ori = []
        for i in range(len(landmarks_next)):
            if i % 2 == 0:
                landmarks_next_ori.append([int(landmarks_next[i]), int(landmarks_next[i + 1])])

        # 是否进行人脸矫正
        if self.faceRotate == True:
            image_rotate, landmarks_rotate = self.face_rotate(img, landmarks_ori)
            image_rotate_next, landmarks_rotate_next = self.face_rotate(img_next, landmarks_next_ori)
        else:
            image_rotate, landmarks_rotate = img, np.array(landmarks_ori)
            image_rotate_next, landmarks_rotate_next = img_next, np.array(landmarks_next_ori)
        image_rotate_shape = image_rotate.shape
        rows = image_rotate_shape[0]  # height(rows) of image
        cols = image_rotate_shape[1]  # width(colums) of image

        # 根据矫正后的人脸特征点选取人脸框位置
        point_x = []
        point_y = []
        for i in range(len(landmarks_rotate)):
            point_x.append(int(landmarks_rotate[i][0]))
            point_y.append(int(landmarks_rotate[i][1]))
        ymin = min(point_y)
        ymax = max(point_y)
        xmin = min(point_x)
        xmax = max(point_x)
        ymin = max(ymin, 0)
        ymax = min(ymax, rows)
        xmin = max(xmin, 0)
        xmax = min(xmax, cols)

        image_rotate_next_shape = image_rotate_next.shape
        rows_next = image_rotate_next_shape[0]  # height(rows) of image
        cols_next = image_rotate_next_shape[1]  # width(colums) of image

        # 根据矫正后的人脸特征点选取人脸框位置
        point_next_x = []
        point_next_y = []
        for i in range(len(landmarks_rotate_next)):
            point_next_x.append(int(landmarks_rotate_next[i][0]))
            point_next_y.append(int(landmarks_rotate_next[i][1]))
        ymin_next = min(point_next_y)
        ymax_next = max(point_next_y)
        xmin_next = min(point_next_x)
        xmax_next = max(point_next_x)
        ymin_next = max(ymin_next, 0)
        ymax_next = min(ymax_next, rows_next)
        xmin_next = max(xmin_next, 0)
        xmax_next = min(xmax_next, cols_next)

        # 生成mask，和图像相同尺寸，然后再裁剪
        mask = np.zeros((rows, cols), dtype='uint8')

        mask_next = np.zeros((rows_next, cols_next), dtype='uint8')
        if self.maskType == 'rect':
            mask[ymin:ymax, xmin:xmax] = 1
            mask_next[ymin_next:ymax_next, xmin_next:xmax_next] = 1
        if self.maskType == 'face':
            landmarks_src_rotate = np.array(landmarks_rotate, dtype=np.int32)
            cv2.fillConvexPoly(mask, cv2.convexHull(landmarks_src_rotate), 1)  # 填充凸多边形

            landmarks_src_rotate_next = np.array(landmarks_rotate_next, dtype=np.int32)
            cv2.fillConvexPoly(mask_next, cv2.convexHull(landmarks_src_rotate_next), 1)  # 填充凸多边形

        # 将人脸和mask向外扩大k倍
        ymin2, ymax2, xmin2, xmax2 = self.face_resize(rows, cols, ymin, ymax, xmin, xmax, k=1.3)
        img_face = image_rotate[ymin2:ymax2, xmin2:xmax2]  # cv的裁剪图像方式，裁剪坐标为[ymin:ymax, xmin:xmax]
        mask_face = mask[ymin2:ymax2, xmin2:xmax2]

        ymin2_next, ymax2_next, xmin2_next, xmax2_next = self.face_resize(rows_next, cols_next, ymin_next, ymax_next,
                                                                          xmin_next, xmax_next, k=1.3)
        img_face_next = image_rotate_next[ymin2_next:ymax2_next,
                        xmin2_next:xmax2_next]  # cv的裁剪图像方式，裁剪坐标为[ymin:ymax, xmin:xmax]
        mask_face_next = mask_next[ymin2_next:ymax2_next, xmin2_next:xmax2_next]
        rel_landmarks = landmarks_rotate - [xmin2, ymin2]
        if self.model_sbi:
            rel_landmarks = landmarks_rotate - [xmin2, ymin2]
            img_face_sbi_true, img_face_sbi_false, mask_face_sbi, replay_info = self.self_blending(
                img_face, rel_landmarks, replay_data=None
            )

            if self.sbi_aug:
                img_face = img_face_sbi_true
            else:    
                img_face = img_face_sbi_false
                label=0
            mask_face = mask_face_sbi


        img_face_aug = img_face

        img_face_aug_next = img_face_next

        if self.spsl:
            BGR_img_face = cv2.resize(img_face_aug, (self.image_size, self.image_size))
            gray_img_face = cv2.cvtColor(BGR_img_face, cv2.COLOR_BGR2GRAY)
            gray_img_face = np.float32(gray_img_face) 
            fft_gray_img_face = np.fft.fft2(gray_img_face)
            ip = np.angle(fft_gray_img_face)  # 相位谱
            recon_Phase0 = np.fft.ifft2(np.exp(1j * ip))
            recon_Phase1 = recon_Phase0
            recon_Phase2 = np.uint8((recon_Phase1/np.max(recon_Phase1)) * 255)
            img_face_b, img_face_g, img_face_r = cv2.split(BGR_img_face)                            # 分离通道
            RGB_img_face = cv2.merge([img_face_b, img_face_g, img_face_r, recon_Phase2])
        else:
            # 图像尺寸和模式调整
            BGR_img_face = cv2.resize(img_face_aug, (self.image_size, self.image_size))
            RGB_img_face = cv2.cvtColor(BGR_img_face, cv2.COLOR_BGR2RGB)
        # 将mask变为图像模式，像素范围在0~255范围
        mask_face = cv2.resize(mask_face, (self.mask_size, self.mask_size))
        mask_face = mask_face.squeeze()

        BGR_img_face_next = cv2.resize(img_face_aug_next, (self.image_size, self.image_size))
        RGB_img_face_next = cv2.cvtColor(BGR_img_face_next, cv2.COLOR_BGR2RGB)
        # 将mask变为图像模式，像素范围在0~255范围
        mask_face_next = cv2.resize(mask_face_next, (self.mask_size, self.mask_size))
        mask_face_next = mask_face_next.squeeze()
        img_face_input = Image.fromarray(np.uint8(RGB_img_face))
        mask_face_input = Image.fromarray(np.uint8(mask_face))

        img_face_input_next = Image.fromarray(np.uint8(RGB_img_face_next))
        mask_face_input_next = Image.fromarray(np.uint8(mask_face_next))
        img_face_input = torch.from_numpy((np.transpose(np.array(img_face_input), (2, 0, 1))).astype(np.float32))
        mask_face_input = torch.from_numpy(np.expand_dims(np.array(mask_face_input), axis=0).astype(np.float32))
        img_face_input_next = torch.from_numpy(
            (np.transpose(np.array(img_face_input_next), (2, 0, 1))).astype(np.float32))
        mask_face_input_next = torch.from_numpy(
            np.expand_dims(np.array(mask_face_input_next), axis=0).astype(np.float32))
        
        return img_face_input, mask_face_input, label, self.domain_ID, path, img_face_input_next, mask_face_input_next, rel_landmarks

        
    def __len__(self):
        """
        返回数据集中图像的数量
        :return:
        """
        return len(self.imgs_path)

    def face_resize(self, rows, cols, ymin, ymax, xmin, xmax, k=1.3):
        """获得k倍人脸"""
        height = ymax - ymin
        width = xmax - xmin
        height1 = round(k * height)
        width1 = round(k * width)
        center_y = round((ymin + ymax) / 2)
        center_x = round((xmin + xmax) / 2)
        ymin = round(center_y - height1 / 2)
        ymax = round(center_y + height1 / 2)
        xmin = round(center_x - width1 / 2)
        xmax = round(center_x + width1 / 2)
        if ymin <= 0:
            ymin = 0
        if ymax >= rows:
            ymax = rows
        if xmin <= 0:
            xmin = 0
        if xmax >= cols:
            xmax = cols
        return ymin, ymax, xmin, xmax

    def face_rotate(self, image, landmarks):
        """人脸对齐矫正"""
        landmarks = np.array(landmarks)
        # rotation angle计算旋转角度
        left_eye_corner = landmarks[36]  # 左眼外角
        right_eye_corner = landmarks[45]  # 右眼外角
        # 计算偏转角
        radian = np.arctan((left_eye_corner[1] - right_eye_corner[1]) / (left_eye_corner[0] - right_eye_corner[0]))
        # print('radian：' + str(radian))
        degree = math.degrees(radian)
        # print('degree：'+str(degree))

        if abs(degree) > 20:
            return image, landmarks
        else:
            # image size after rotating
            height, width, _ = image.shape
            cos = math.cos(radian)
            sin = math.sin(radian)

            new_w = width * abs(cos) + height * abs(sin)
            new_h = width * abs(sin) + height * abs(cos)
            if np.isnan(new_w):
                return image, landmarks
            elif np.isnan(new_h):
                return image, landmarks
            else:
                new_w = int(new_w)
                new_h = int(new_h)

                # translation
                Tx = new_w // 2 - width // 2
                Ty = new_h // 2 - height // 2

                # affine matrix
                M = np.array([[cos, sin, (1 - cos) * width / 2. - sin * height / 2. + Tx],
                              [-sin, cos, sin * width / 2. + (1 - cos) * height / 2. + Ty]])

                image_rotate = cv2.warpAffine(image, M, (new_w, new_h), borderValue=(255, 255, 255))

                landmarks = np.concatenate([landmarks, np.ones((landmarks.shape[0], 1))], axis=1)
                landmarks_rotate = np.dot(M, landmarks.T).T
                return image_rotate, landmarks_rotate
