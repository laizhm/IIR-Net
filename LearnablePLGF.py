import os
# import cv2
import time
import torch
import torchvision
import numpy as np

from glob import glob
from tqdm import tqdm
import torch.nn.functional as F
from PIL import Image


class LearnablePLGF(nn.Module):
    def __init__(self, kernel_size=5):
        super(LearnablePLGF, self).__init__()
        # 支持的卷积核尺寸为3、5、7
        self.kernel_size = kernel_size
        self.sigmoid = torch.nn.Sigmoid()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=3,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=6,
                out_channels=6,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=12,
                out_channels=12,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(12),
            nn.ReLU(inplace=True)
        )
        filter_x = self.get_filter_x(self.kernel_size)
        filter_y = self.get_filter_y(self.kernel_size)
        self.padding_x = filter_x.shape[0] // 2
        self.padding_y = filter_y.shape[0] // 2
        filter_x = torch.Tensor(np.expand_dims(self.get_filter_x(self.kernel_size), axis=(0, 1))).float()
        filter_y = torch.Tensor(np.expand_dims(self.get_filter_y(self.kernel_size), axis=(0, 1))).float()
        self.kernel_y = torch.nn.Parameter(torch.zeros([32 * 3, 1, 3, 3], requires_grad=True) + filter_x)
        self.kernel_x = torch.nn.Parameter(torch.zeros([32 * 3, 1, 3, 3], requires_grad=True) + filter_y)

    def get_filter_x(self, kernel_size):
        if kernel_size == 3:
            filter_x = np.array([[-1 / (2 * np.sqrt(2)), 0, 1 / (2 * np.sqrt(2))],
                                 [-1, 0, 1],
                                 [-1 / (2 * np.sqrt(2)), 0, 1 / (2 * np.sqrt(2))]], dtype=np.float32)
        elif kernel_size == 5:
            filter_x = np.array([
                [np.cos(6 / 8 * np.pi) / 8, np.cos(5 / 8 * np.pi) / 5, np.cos(4 / 8 * np.pi) / 4,
                 np.cos(3 / 8 * np.pi) / 5, np.cos(2 / 8 * np.pi) / 8],
                [np.cos(7 / 8 * np.pi) / 5, -1 / (2 * np.sqrt(2)), 0, 1 / (2 * np.sqrt(2)),
                 np.cos(1 / 8 * np.pi) / 5],
                [np.cos(8 / 8 * np.pi) / 4, -1, 0, 1, np.cos(0 / 8 * np.pi) / 4],
                [np.cos(9 / 8 * np.pi) / 5, -1 / (2 * np.sqrt(2)), 0, 1 / (2 * np.sqrt(2)),
                 np.cos(15 / 8 * np.pi) / 5],
                [np.cos(10 / 8 * np.pi) / 8, np.cos(11 / 8 * np.pi) / 5, np.cos(12 / 8 * np.pi) / 4,
                 np.cos(13 / 8 * np.pi) / 5,
                 np.cos(14 / 8 * np.pi) / 8]], dtype=np.float32)
        elif kernel_size == 7:
            filter_x = np.array([[np.cos(9 / 12 * np.pi) / 18, np.cos(8 / 12 * np.pi) / 13, np.cos(7 / 12 * np.pi) / 10,
                                  np.cos(6 / 12 * np.pi) / 9,
                                  np.cos(5 / 12 * np.pi) / 10, np.cos(4 / 12 * np.pi) / 13,
                                  np.cos(3 / 12 * np.pi) / 18],
                                 [np.cos(10 / 12 * np.pi) / 13, np.cos(6 / 8 * np.pi) / 8, np.cos(5 / 8 * np.pi) / 5,
                                  np.cos(4 / 8 * np.pi) / 4,
                                  np.cos(3 / 8 * np.pi) / 5, np.cos(2 / 8 * np.pi) / 8, np.cos(2 / 12 * np.pi) / 13],
                                 [np.cos(11 / 12 * np.pi) / 10, np.cos(7 / 8 * np.pi) / 5, -1 / (2 * np.sqrt(2)), 0,
                                  1 / (2 * np.sqrt(2)), np.cos(1 / 8 * np.pi) / 5, np.cos(1 / 12 * np.pi) / 10],
                                 [np.cos(12 / 12 * np.pi) / 9, np.cos(8 / 8 * np.pi) / 4, -1, 0, 1,
                                  np.cos(0 / 8 * np.pi) / 4, np.cos(0 / 12 * np.pi) / 9],
                                 [np.cos(13 / 12 * np.pi) / 10, np.cos(9 / 8 * np.pi) / 5, -1 / (2 * np.sqrt(2)), 0,
                                  1 / (2 * np.sqrt(2)), np.cos(15 / 8 * np.pi) / 5, np.cos(23 / 12 * np.pi) / 10],
                                 [np.cos(14 / 12 * np.pi) / 13, np.cos(10 / 8 * np.pi) / 8, np.cos(11 / 8 * np.pi) / 5,
                                  np.cos(12 / 8 * np.pi) / 4,
                                  np.cos(13 / 8 * np.pi) / 5, np.cos(14 / 8 * np.pi) / 8, np.cos(22 / 12 * np.pi) / 13],
                                 [np.cos(15 / 12 * np.pi) / 18, np.cos(16 / 12 * np.pi) / 13,
                                  np.cos(17 / 12 * np.pi) / 10, np.cos(18 / 12 * np.pi) / 9,
                                  np.cos(19 / 12 * np.pi) / 10, np.cos(20 / 12 * np.pi) / 13,
                                  np.cos(21 / 12 * np.pi) / 18]], dtype=np.float32)
        else:
            raise ValueError("error kernel size %d" % kernel_size)

        # statck the filters
        # filter_x = [filter_x, filter_x, filter_x]  # (3,k,k)
        # filter_x = torch.FloatTensor(filter_x)
        return filter_x

    def get_filter_y(self, kernel_size):
        if kernel_size == 3:
            filter_y = np.array([[1 / (2 * np.sqrt(2)), 1, 1 / (2 * np.sqrt(2))],
                                 [0, 0, 0],
                                 [-1 / (2 * np.sqrt(2)), -1, -1 / (2 * np.sqrt(2))]], dtype=np.float32)

        elif kernel_size == 5:
            filter_y = np.array([
                [np.sin(6 / 8 * np.pi) / 8, np.sin(5 / 8 * np.pi) / 5, np.sin(4 / 8 * np.pi) / 4,
                 np.sin(3 / 8 * np.pi) / 5, np.sin(2 / 8 * np.pi) / 8],
                [np.sin(7 / 8 * np.pi) / 5, 1 / (2 * np.sqrt(2)), 1, 1 / (2 * np.sqrt(2)), np.sin(1 / 8 * np.pi) / 5],
                [np.sin(8 / 8 * np.pi) / 4, 0, 0, 0, np.sin(0 / 8 * np.pi) / 4],
                [np.sin(9 / 8 * np.pi) / 5, -1 / (2 * np.sqrt(2)), -1, -1 / (2 * np.sqrt(2)),
                 np.sin(15 / 8 * np.pi) / 5],
                [np.sin(10 / 8 * np.pi) / 8, np.sin(11 / 8 * np.pi) / 5, np.sin(12 / 8 * np.pi) / 4,
                 np.sin(13 / 8 * np.pi) / 5,
                 np.sin(14 / 8 * np.pi) / 8]], dtype=np.float32)
        elif kernel_size == 7:
            filter_y = np.array([[np.sin(9 / 12 * np.pi) / 18, np.sin(8 / 12 * np.pi) / 13, np.sin(7 / 12 * np.pi) / 10,
                                  np.sin(6 / 12 * np.pi) / 9,
                                  np.sin(5 / 12 * np.pi) / 10, np.sin(4 / 12 * np.pi) / 13,
                                  np.sin(3 / 12 * np.pi) / 18],
                                 [np.sin(10 / 12 * np.pi) / 13, np.sin(6 / 8 * np.pi) / 8, np.sin(5 / 8 * np.pi) / 5,
                                  np.sin(4 / 8 * np.pi) / 4,
                                  np.sin(3 / 8 * np.pi) / 5, np.sin(2 / 8 * np.pi) / 8, np.sin(2 / 12 * np.pi) / 13],
                                 [np.sin(11 / 12 * np.pi) / 10, np.sin(7 / 8 * np.pi) / 5, 1 / (2 * np.sqrt(2)), 1,
                                  1 / (2 * np.sqrt(2)), np.sin(1 / 8 * np.pi) / 5, np.sin(1 / 12 * np.pi) / 10],
                                 [np.sin(12 / 12 * np.pi) / 9, np.sin(8 / 8 * np.pi) / 4, 0, 0, 0,
                                  np.sin(0 / 8 * np.pi) / 4, np.sin(0 / 12 * np.pi) / 9],
                                 [np.sin(13 / 12 * np.pi) / 10, np.sin(9 / 8 * np.pi) / 5, -1 / (2 * np.sqrt(2)), -1,
                                  -1 / (2 * np.sqrt(2)), np.sin(15 / 8 * np.pi) / 5, np.sin(23 / 12 * np.pi) / 10],
                                 [np.sin(14 / 12 * np.pi) / 13, np.sin(10 / 8 * np.pi) / 8, np.sin(11 / 8 * np.pi) / 5,
                                  np.sin(12 / 8 * np.pi) / 4,
                                  np.sin(13 / 8 * np.pi) / 5, np.sin(14 / 8 * np.pi) / 8, np.sin(22 / 12 * np.pi) / 13],
                                 [np.sin(15 / 12 * np.pi) / 18, np.sin(16 / 12 * np.pi) / 13,
                                  np.sin(17 / 12 * np.pi) / 10, np.sin(18 / 12 * np.pi) / 9,
                                  np.sin(19 / 12 * np.pi) / 10, np.sin(20 / 12 * np.pi) / 13,
                                  np.sin(21 / 12 * np.pi) / 18]], dtype=np.float32)

        else:
            raise ValueError("error kernel size %d" % kernel_size)

        # statck the filters
        # filter_y = [filter_y, filter_y, filter_y]  # (3,k,k)
        # filter_y = torch.FloatTensor(filter_y)
        return filter_y

    def normalization(self, data_in, max_output=255, min_output=0):
        max_value = torch.max(data_in)
        min_value = torch.min(data_in)
        out = ((data_in - min_value) / (max_value - min_value)) * (max_output - min_output) + min_output
        return out

    def forward(self, image):
        """
        image为tensor数据，维度为(N,C,H,W)
        """
        N, C, H, W = image.shape
        # 输入需要是3通道RGB图像
        if C != 3:
            raise ValueError("image_channel = {}, which is not match requirements".format(C))
        image = torch.where(image > 2.0, image, 2.0)
        image = image.reshape(1, -1, H, W)
        plgf_x = F.conv2d(input=image, weight=self.kernel_x[:N * C], padding=self.padding_x, groups=N*3)  # [1,1,H,W]
        plgf_y = F.conv2d(input=image, weight=self.kernel_y[:N * C], padding=self.padding_y, groups=N*3)  # [1,1,H,W]
        plgf_img = torch.atan(torch.sqrt(
            torch.pow(torch.div(plgf_x, image + 0.0001), 2) + torch.pow(torch.div(plgf_y, image + 0.0001), 2)))  # [1,1,H,W]
        plgf_img = self.normalization(plgf_img, 1, 0)                   # [H,W]
        plgf_img = self.sigmoid(plgf_img)                               # [H,W]
        plgf_img = plgf_img.reshape(N, C, H, W)
        plgf1 = self.conv1(plgf_img)
        plgf1 = torch.cat([plgf_img, plgf1], dim=1)
        plgf2 = self.conv2(plgf1)
        plgf2 = torch.cat([plgf1, plgf2], dim=1)
        plgf3 = self.conv3(plgf2)
        plgf3 = torch.cat([plgf2, plgf3], dim=1)
        return plgf3


class OriPLGF(nn.Module):
    def __init__(self, kernel_size=5):
        super(OriPLGF, self).__init__()
        # 支持的卷积核尺寸为3、5、7
        self.kernel_size = kernel_size
        self.sigmoid = torch.nn.Sigmoid()

    def get_filter_x(self, kernel_size):
        if kernel_size == 3:
            filter_x = torch.tensor([[-1 / (2 * torch.sqrt(torch.tensor(2.0))), 0, 1 / (2 * torch.sqrt(torch.tensor(2.0)))],
                                     [-1, 0, 1],
                                     [-1 / (2 * torch.sqrt(torch.tensor(2.0))), 0, 1 / (2 * torch.sqrt(torch.tensor(2.0)))]], dtype=torch.float32)
        elif kernel_size == 5:
            filter_x = torch.tensor([
                [torch.cos(torch.tensor(6 / 8) * torch.pi) / 8, torch.cos(torch.tensor(5 / 8) * torch.pi) / 5, torch.cos(torch.tensor(4 / 8) * torch.pi) / 4,
                 torch.cos(torch.tensor(3 / 8) * torch.pi) / 5, torch.cos(torch.tensor(2 / 8) * torch.pi) / 8],
                [torch.cos(torch.tensor(7 / 8) * torch.pi) / 5, -1 / (2 * torch.sqrt(torch.tensor(2.0))), 0, 1 / (2 * torch.sqrt(torch.tensor(2.0))),
                 torch.cos(torch.tensor(1 / 8) * torch.pi) / 5],
                [torch.cos(torch.tensor(8 / 8) * torch.pi) / 4, -1, 0, 1, torch.cos(torch.tensor(0 / 8) * torch.pi) / 4],
                [torch.cos(torch.tensor(9 / 8) * torch.pi) / 5, -1 / (2 * torch.sqrt(torch.tensor(2.0))), 0, 1 / (2 * torch.sqrt(torch.tensor(2.0))),
                 torch.cos(torch.tensor(15 / 8) * torch.pi) / 5],
                [torch.cos(torch.tensor(10 / 8) * torch.pi) / 8, torch.cos(torch.tensor(11 / 8) * torch.pi) / 5, torch.cos(torch.tensor(12 / 8) * torch.pi) / 4,
                 torch.cos(torch.tensor(13 / 8) * torch.pi) / 5,
                 torch.cos(torch.tensor(14 / 8) * torch.pi) / 8]], dtype=torch.float32)
        elif kernel_size == 7:
            filter_x = torch.tensor([[torch.cos(torch.tensor(9 / 12) * torch.pi) / 18, torch.cos(torch.tensor(8 / 12) * torch.pi) / 13, torch.cos(torch.tensor(7 / 12) * torch.pi) / 10,
                                      torch.cos(torch.tensor(6 / 12) * torch.pi) / 9,
                                      torch.cos(torch.tensor(5 / 12) * torch.pi) / 10, torch.cos(torch.tensor(4 / 12) * torch.pi) / 13,
                                      torch.cos(torch.tensor(3 / 12) * torch.pi) / 18],
                                     [torch.cos(torch.tensor(10 / 12) * torch.pi) / 13, torch.cos(torch.tensor(6 / 8) * torch.pi) / 8, torch.cos(torch.tensor(5 / 8) * torch.pi) / 5,
                                      torch.cos(torch.tensor(4 / 8) * torch.pi) / 4,
                                      torch.cos(torch.tensor(3 / 8) * torch.pi) / 5, torch.cos(torch.tensor(2 / 8) * torch.pi) / 8, torch.cos(torch.tensor(2 / 12) * torch.pi) / 13],
                                     [torch.cos(torch.tensor(11 / 12) * torch.pi) / 10, torch.cos(torch.tensor(7 / 8) * torch.pi) / 5, -1 / (2 * torch.sqrt(torch.tensor(2.0))), 0,
                                      1 / (2 * torch.sqrt(torch.tensor(2.0))), torch.cos(torch.tensor(1 / 8) * torch.pi) / 5, torch.cos(torch.tensor(1 / 12) * torch.pi) / 10],
                                     [torch.cos(torch.tensor(12 / 12) * torch.pi) / 9, torch.cos(torch.tensor(8 / 8) * torch.pi) / 4, -1, 0, 1,
                                      torch.cos(torch.tensor(0 / 8) * torch.pi) / 4, torch.cos(torch.tensor(0 / 12) * torch.pi) / 9],
                                     [torch.cos(torch.tensor(13 / 12) * torch.pi) / 10, torch.cos(torch.tensor(9 / 8) * torch.pi) / 5, -1 / (2 * torch.sqrt(torch.tensor(2.0))), 0,
                                      1 / (2 * torch.sqrt(torch.tensor(2.0))), torch.cos(torch.tensor(15 / 8) * torch.pi) / 5, torch.cos(torch.tensor(23 / 12) * torch.pi) / 10],
                                     [torch.cos(torch.tensor(14 / 12) * torch.pi) / 13, torch.cos(torch.tensor(10 / 8) * torch.pi) / 8, torch.cos(torch.tensor(11 / 8) * torch.pi) / 5,
                                      torch.cos(torch.tensor(12 / 8) * torch.pi) / 4,
                                      torch.cos(torch.tensor(13 / 8) * torch.pi) / 5, torch.cos(torch.tensor(14 / 8) * torch.pi) / 8, torch.cos(torch.tensor(22 / 12) * torch.pi) / 13],
                                     [torch.cos(torch.tensor(15 / 12) * torch.pi) / 18, torch.cos(torch.tensor(16 / 12) * torch.pi) / 13,
                                      torch.cos(torch.tensor(17 / 12) * torch.pi) / 10, torch.cos(torch.tensor(18 / 12) * torch.pi) / 9,
                                      torch.cos(torch.tensor(19 / 12) * torch.pi) / 10, torch.cos(torch.tensor(20 / 12) * torch.pi) / 13,
                                      torch.cos(torch.tensor(21 / 12) * torch.pi) / 18]], dtype=torch.float32)
        else:
            raise ValueError("error kernel size %d" % kernel_size)

        return filter_x

    def get_filter_y(self, kernel_size):
        if kernel_size == 3:
            filter_y = torch.tensor([[1 / (2 * torch.sqrt(torch.tensor(2.0))), 1, 1 / (2 * torch.sqrt(torch.tensor(2.0)))],
                                     [0, 0, 0],
                                     [-1 / (2 * torch.sqrt(torch.tensor(2.0))), -1, -1 / (2 * torch.sqrt(torch.tensor(2.0)))]], dtype=torch.float32)

        elif kernel_size == 5:
            filter_y = torch.tensor([
                [torch.sin(torch.tensor(6 / 8) * torch.pi) / 8, torch.sin(torch.tensor(5 / 8) * torch.pi) / 5, torch.sin(torch.tensor(4 / 8) * torch.pi) / 4,
                 torch.sin(torch.tensor(3 / 8) * torch.pi) / 5, torch.sin(torch.tensor(2 / 8) * torch.pi) / 8],
                [torch.sin(torch.tensor(7 / 8) * torch.pi) / 5, 1 / (2 * torch.sqrt(torch.tensor(2.0))), 1, 1 / (2 * torch.sqrt(torch.tensor(2.0))), torch.sin(torch.tensor(1 / 8) * torch.pi) / 5],
                [torch.sin(torch.tensor(8 / 8) * torch.pi) / 4, 0, 0, 0, torch.sin(torch.tensor(0 / 8) * torch.pi) / 4],
                [torch.sin(torch.tensor(9 / 8) * torch.pi) / 5, -1 / (2 * torch.sqrt(torch.tensor(2.0))), -1, -1 / (2 * torch.sqrt(torch.tensor(2.0))),
                 torch.sin(torch.tensor(15 / 8) * torch.pi) / 5],
                [torch.sin(torch.tensor(10 / 8) * torch.pi) / 8, torch.sin(torch.tensor(11 / 8) * torch.pi) / 5, torch.sin(torch.tensor(12 / 8) * torch.pi) / 4,
                 torch.sin(torch.tensor(13 / 8) * torch.pi) / 5,
                 torch.sin(torch.tensor(14 / 8) * torch.pi) / 8]], dtype=torch.float32)
        elif kernel_size == 7:
            filter_y = torch.tensor([[torch.sin(torch.tensor(9 / 12) * torch.pi) / 18, torch.sin(torch.tensor(8 / 12) * torch.pi) / 13, torch.sin(torch.tensor(7 / 12) * torch.pi) / 10,
                                      torch.sin(torch.tensor(6 / 12) * torch.pi) / 9,
                                      torch.sin(torch.tensor(5 / 12) * torch.pi) / 10, torch.sin(torch.tensor(4 / 12) * torch.pi) / 13,
                                      torch.sin(torch.tensor(3 / 12) * torch.pi) / 18],
                                     [torch.sin(torch.tensor(10 / 12) * torch.pi) / 13, torch.sin(torch.tensor(6 / 8) * torch.pi) / 8, torch.sin(torch.tensor(5 / 8) * torch.pi) / 5,
                                      torch.sin(torch.tensor(4 / 8) * torch.pi) / 4,
                                      torch.sin(torch.tensor(3 / 8) * torch.pi) / 5, torch.sin(torch.tensor(2 / 8) * torch.pi) / 8, torch.sin(torch.tensor(2 / 12) * torch.pi) / 13],
                                     [torch.sin(torch.tensor(11 / 12) * torch.pi) / 10, torch.sin(torch.tensor(7 / 8) * torch.pi) / 5, 1 / (2 * torch.sqrt(torch.tensor(2.0))), 1,
                                      1 / (2 * torch.sqrt(torch.tensor(2.0))), torch.sin(torch.tensor(1 / 8) * torch.pi) / 5, torch.sin(torch.tensor(1 / 12) * torch.pi) / 10],
                                     [torch.sin(torch.tensor(12 / 12) * torch.pi) / 9, torch.sin(torch.tensor(8 / 8) * torch.pi) / 4, 0, 0, 0,
                                      torch.sin(torch.tensor(0 / 8) * torch.pi) / 4, torch.sin(torch.tensor(0 / 12) * torch.pi) / 9],
                                     [torch.sin(torch.tensor(13 / 12) * torch.pi) / 10, torch.sin(torch.tensor(9 / 8) * torch.pi) / 5, -1 / (2 * torch.sqrt(torch.tensor(2.0))), -1,
                                      -1 / (2 * torch.sqrt(torch.tensor(2.0))), torch.sin(torch.tensor(15 / 8) * torch.pi) / 5, torch.sin(torch.tensor(23 / 12) * torch.pi) / 10],
                                     [torch.sin(torch.tensor(14 / 12) * torch.pi) / 13, torch.sin(torch.tensor(10 / 8) * torch.pi) / 8, torch.sin(torch.tensor(11 / 8) * torch.pi) / 5,
                                      torch.sin(torch.tensor(12 / 8) * torch.pi) / 4,
                                      torch.sin(torch.tensor(13 / 8) * torch.pi) / 5, torch.sin(torch.tensor(14 / 8) * torch.pi) / 8, torch.sin(torch.tensor(22 / 12) * torch.pi) / 13],
                                     [torch.sin(torch.tensor(15 / 12) * torch.pi) / 18, torch.sin(torch.tensor(16 / 12) * torch.pi) / 13,
                                      torch.sin(torch.tensor(17 / 12) * torch.pi) / 10, torch.sin(torch.tensor(18 / 12) * torch.pi) / 9,
                                      torch.sin(torch.tensor(19 / 12) * torch.pi) / 10, torch.sin(torch.tensor(20 / 12) * torch.pi) / 13,
                                      torch.sin(torch.tensor(21 / 12) * torch.pi) / 18]], dtype=torch.float32)

        else:
            raise ValueError("error kernel size %d" % kernel_size)
        return filter_y

    def plgf_conv(self, image, filter):
        padding = filter.shape[0] // 2

        image_input = image.float().unsqueeze(0).unsqueeze(0)
        kernel = filter.float().unsqueeze(0).unsqueeze(0)

        kernel = torch.nn.Parameter(kernel, requires_grad=False)
        output = F.conv2d(input=image_input, weight=kernel, padding=padding)
        return output

    def normalization(self, data_in, max_output=255, min_output=0):
        max_value = torch.max(data_in)
        min_value = torch.min(data_in)
        out = ((data_in - min_value) / (max_value - min_value)) * (max_output - min_output) + min_output
        return out

    def forward(self, image):
        """
        image为tensor数据，维度为(N,C,H,W)
        """
        N, C, H, W = image.shape
        # 输入需要是3通道RGB图像
        if C != 3:
            raise ValueError("image_channel = {}, which is not match requirements".format(C))

        plgf = image.detach().clone()
        for i in range(N):
            image_i = image[i, :, :, :]  # [C,H,W]
            image_i = torch.where(image_i > 2, image_i, 2)  # The source code here makes the pixel value must be greater than or equal to 2
            image_r, image_g, image_b = image_i[0, :, :], image_i[1, :, :], image_i[2, :, :]
            image_channel_list = [image_r, image_g, image_b]
            for j, img in enumerate(image_channel_list):
                filter_x, filter_y = self.get_filter_x(self.kernel_size), self.get_filter_y(self.kernel_size)

                plgf_x = self.plgf_conv(img, filter_x.to(image.device))  # shape of plgf_x/y is (1,C,H,W) → (1,1,256,256)
                plgf_y = self.plgf_conv(img, filter_y.to(image.device))

                plgf_image = torch.atan(torch.sqrt(((plgf_x / (img + 0.0001)) ** 2) + ((plgf_y / (img + 0.0001)) ** 2)))
                plgf_norm = self.normalization(plgf_image, 255, 1)
                plgf[i, j, :, :] = plgf_norm

        return plgf