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
from tqdm import tqdm


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
    def __init__(self, rootpth='', des_size=(512, 512), mode='train', ):

        '''

        :param rootpth: 根目录
        :param re_size: 数据同一resize到这个尺寸再后处理
        :param crop_size: 剪切
        :param erase: 遮罩比例
        :param mode: train/val/test
        '''

        self.root_path = rootpth
        self.des_size = des_size
        self.mode = mode

        # 处理对应标签
        assert (mode == 'train' or mode == 'val' or mode == 'test')
        labels = ["epidural", "intraparenchymal", "intraventricular",
                  "subarachnoid", "subdural", "any"]
        self.label_ranks = {}
        for i in range(len(labels)):
            self.label_ranks[labels[i]] = i

        lines = open(osp.join(rootpth, 'ClsName2id.txt'), 'r', encoding='utf-8').read().rstrip().split('\n')

        self.catetory2idx = {}
        for line in lines:
            line_list = line.strip().split(':')
            self.catetory2idx[line_list[0]] = int(line_list[2])-1

        # 读取文件名称
        self.file_names = []
        for root,dirs,names in os.walk(osp.join(rootpth, mode)):
            for name in names:
                self.file_names.append(osp.join(root,name))

        # 随机选择一小部分数据做测试
        # self.file_names = random.choices(self.file_names[:200],k=5000)

        # 确定分隔符号
        self.split_char = '\\' if '\\' in self.file_names[0] else '/'

        # totensor 转换n
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def data_loader(self, fname):
        return pydicom.dcmread(fname).pixel_array

    def prepare_label(self):
        all_true_labels = {}
        import csv
        with open(osp.join(self.root_path, 'stage_1_sample_submission.csv'), 'r') as fp:
            csv_reader = csv.reader(fp, delimiter=',')
            next(csv_reader, None)
            for row in csv_reader:
                id = row[0].split('_')[0]
                label_id = self.label_ranks[row[0].split('_')[1]]
                if id not in all_true_labels:
                    all_true_labels[id][label_id] = [0] * 6
                all_true_labels[id][label_id] = int(row[1])
        return all_true_labels

    def __getitem__(self, idx):
        name = self.file_names[idx]
        category = name.split(self.split_char)[-2]
        cate_int = self.catetory2idx[category]
        img = Image.open(name)
        img = img.resize(self.des_size,Image.BILINEAR)
        return self.to_tensor(img), cate_int

    def __len__(self):
        return len(self.file_names)


class RSDataset_test(Dataset):
    def __init__(self, rootpth='', des_size=(512, 512), mode='train', ):

        '''

        :param rootpth: 根目录
        :param re_size: 数据同一resize到这个尺寸再后处理
        :param crop_size: 剪切
        :param erase: 遮罩比例
        :param mode: train/val/test
        '''

        self.des_size = des_size
        self.mode = mode

        # 处理对应标签
        assert (mode=='train' or mode=='val' or mode=='test')
        lines = open(osp.join(rootpth,'ClsName2id.txt'),'r',encoding='utf-8').read().rstrip().split('\n')
        self.catetory2idx = {}
        for line in lines:
            line_list = line.strip().split(':')
            self.catetory2idx[line_list[0]] = int(line_list[2])-1

        # 读取文件名称
        self.file_names = []
        for root, dirs, names in os.walk(osp.join(rootpth, mode)):
            for name in names:
                self.file_names.append(osp.join(root,name))

        # 随机选择一小部分数据做测试
        # self.file_names = random.choices(self.file_names[:200],k=5000)

        # 确定分隔符号
        self.split_char = '\\' if '\\' in self.file_names[0] else '/'

        # totensor 转换
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __getitem__(self, idx):
        name = self.file_names[idx]
        category = name.split(self.split_char)[-2]
        # cate_int = self.catetory2idx[category]
        img = Image.open(name)
        img = img.resize(self.des_size,Image.BILINEAR)
        return self.to_tensor(img), 0

    def __len__(self):
        return len(self.file_names)


if __name__ == '__main__':
    data_understanding()