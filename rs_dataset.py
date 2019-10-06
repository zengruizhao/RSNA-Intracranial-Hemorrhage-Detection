# -*- coding: utf-8 -*-
"""
@File    : rs_dataset.py
@Time    : 2019/6/22 10:57
@Author  : Parker
@Email   : now_cherish@163.com
@Software: PyCharm
@Des     : data set
"""

import csv
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pydicom
import os.path as osp
import os
from PIL import Image
import numpy as np
import random
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from skimage.morphology import remove_small_holes, remove_small_objects
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu

def data_understanding():
    labels = prepare_label()
    s, ss = {}, {}
    for key, one in tqdm(zip(list(labels.keys()), list(labels.values()))):
        lb = int("".join(map(str, one)), 2)
        if lb not in s.keys():
            s[lb] = []
        s[lb].append(key)

    for one in labels.values():
        for idx, t in enumerate(one):
            if idx not in ss.keys():
                ss[idx] = 0
            if t == 1:
                ss[idx] += 1
    for one in s.keys():
        print(bin(one)[2:].zfill(6), len(s[one]))


def prepare_label():
    labels = ["epidural", "intraparenchymal", "intraventricular",
              "subarachnoid", "subdural", "any"]
    label_ranks = {}
    for i in range(len(labels)):
        label_ranks[labels[i]] = i
    all_true_labels = {}

    with open(osp.join('/media/tiger/zzr/rsna/stage_1_train.csv'), 'r') as fp:
        csv_reader = csv.reader(fp, delimiter=',')
        next(csv_reader, None)
        print('processing data ...')
        for row in tqdm(csv_reader):
            id = "_".join(row[0].split('_')[:2])
            label_id = label_ranks[row[0].split('_')[2]]
            if id not in all_true_labels:
                all_true_labels[id] = [0] * 6
            all_true_labels[id][label_id] = int(row[1])

    return all_true_labels


class RSDataset(Dataset):
    def __init__(self, rootpth='/media/tiger/zzr/rsna', des_size=(512, 512), mode='train'):
        """
        :param rootpth: 根目录
        :param re_size: 数据同一resize到这个尺寸再后处理
        :param crop_size: 剪切
        :param erase: 遮罩比例
        :param mode: train/val/test
        """
        self.root_path = rootpth
        self.des_size = des_size
        self.mode = mode
        self.name = None

        # 处理对应标签
        assert (mode == 'train' or mode == 'val' or mode == 'test')
        labels = ["epidural", "intraparenchymal", "intraventricular",
                  "subarachnoid", "subdural", "any"]
        self.label_ranks = {}
        for i in range(len(labels)):
            self.label_ranks[labels[i]] = i
        self.labels = self.prepare_label()

        # 读取文件名称
        self.file_names = []
        for root,dirs,names in os.walk(osp.join(rootpth, mode)):
            for name in names:
                if name == 'ID_6431af929.dcm':
                    continue
                self.file_names.append(osp.join(root,name))

        # 确定分隔符号
        self.split_char = '\\' if '\\' in self.file_names[0] else '/'

        # totensor 转换n
        self.to_tensor = transforms.Compose([ # 32.98408291578699 33.70147134726827
            transforms.ToTensor(),
            transforms.Normalize(32.98408291578699, 33.70147134726827)
        ])

    def data_loader(self, fname):
        """
        load data
        :param fname:
        :return:
        """
        ds = pydicom.dcmread(fname)
        try:
            windowCenter = int(ds.WindowCenter[0])
            windowWidth = int(ds.WindowWidth[0])
        except:
            windowCenter = int(ds.WindowCenter)
            windowWidth = int(ds.WindowWidth)
        intercept = ds.RescaleIntercept
        slope = ds.RescaleSlope
        data = ds.pixel_array
        data = np.clip(data * slope + intercept, windowCenter - windowWidth / 2, windowCenter + windowWidth / 2).astype(np.float32)
        data = self.preprocess(data)
        return data

    def preprocess(self, data):
        """
        otsu threshold
        :param data:
        :return:
        """
        try:
            thres = threshold_otsu(data)
        except:
            thres = np.min(data)

        data1 = data > thres
        data1 = remove_small_objects(data1)
        label_data = label(data1)
        props = regionprops(label_data)
        area = 0
        bbox = (0, 0, np.shape(data)[0], np.shape(data)[1])
        for idx, i in enumerate(props):
            if i.area > area:
                area = i.area
                bbox = i.bbox

        data1 = data[bbox[0]:bbox[2]+1, bbox[1]:bbox[-1]+1]

        return data1

    def prepare_label(self):
        all_true_labels = {}
        import csv
        with open(osp.join(self.root_path, 'stage_1_train.csv'), 'r') as fp:
            csv_reader = csv.reader(fp, delimiter=',')
            next(csv_reader, None)
            for row in tqdm(csv_reader):
                id = "_".join(row[0].split('_')[:2])
                label_id = self.label_ranks[row[0].split('_')[2]]
                if id not in all_true_labels:
                    all_true_labels[id] = [0] * 6
                all_true_labels[id][label_id] = float(row[1])

        return all_true_labels

    def __getitem__(self, idx):
        self.name = self.file_names[idx]
        category = self.labels[self.name.split(self.split_char)[-1].split('.')[0]]
        img = cv2.resize(self.data_loader(self.name), dsize=self.des_size, interpolation=cv2.INTER_LINEAR)
        # plt.imshow(img)
        # plt.show()
        return self.to_tensor(img), torch.tensor(category)

    def __len__(self):
        return len(self.file_names)

    def calculateMeanStd(self, idx):
        """

        :param idx:
        :return:
        """
        self.name = self.file_names[idx]
        img = self.data_loader(self.name)

        return np.mean(img), np.std(img)


class RSDataset_test(RSDataset):
    def __init__(self, rootpth='/media/tiger/zzr/rsna', des_size=(512, 512), mode='test'):
        super().__init__()
        # 读取文件名称
        self.file_names = []
        for root, dirs, names in os.walk(osp.join(rootpth, mode)):
            for name in names:
                self.file_names.append(osp.join(root, name))

    def __getitem__(self, idx):
        self.name = self.file_names[idx]
        img = cv2.resize(self.data_loader(self.name), dsize=self.des_size, interpolation=cv2.INTER_LINEAR)
        return self.to_tensor(img), self.name.split(self.split_char)[-1].split('.')[0]

    def __len__(self):
        return len(self.file_names)


if __name__ == '__main__':
    data = RSDataset_test()
    for i in tqdm(range(len(data))):
        a, b = data.__getitem__(i)
        print(data.name)
        print(b)

    # mean, std = 0, 0
    # for i in tqdm(range(len(data))):
    #     u, d = data.calculateMeanStd(i)
    #     u /= len(data)
    #     d /= len(data)
    #     mean += u
    #     std += d
    #
    # print(mean, std)
