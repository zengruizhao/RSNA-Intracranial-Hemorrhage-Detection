# -*- coding: utf-8 -*-
"""
@File    : trian_res34.py
@Time    : 2019/6/23 15:40
@Author  : Parker
@Email   : now_cherish@163.com
@Software: PyCharm
@Des     : 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from tensorboardX import SummaryWriter

import numpy as np
import time
import datetime
import argparse
import os
import os.path as osp

from rs_dataset import RSDataset
from get_logger import get_logger
from res_network import Resnet18, Resnet34


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--epoch', type=int, default=15)
    parse.add_argument('--schedule_step', type=int, default=2)

    parse.add_argument('--batch_size', type=int, default=48)
    parse.add_argument('--test_batch_size', type=int, default=256)
    parse.add_argument('--num_workers', type=int, default=16)

    parse.add_argument('--eval_fre', type=int, default=1)
    parse.add_argument('--msg_fre', type=int, default=10)
    parse.add_argument('--save_fre', type=int, default=2)

    parse.add_argument('--name', type=str, default='res34_baseline',
                       help='unique out file name of this task include log/model_out/tensorboard log')
    parse.add_argument('--data_dir', type=str, default='/home/tiger/projects/rscup2019_classifier/data')
    parse.add_argument('--log_dir', type=str, default='./logs')
    parse.add_argument('--tensorboard_dir', type=str, default='./tensorboard')
    parse.add_argument('--model_out_dir', type=str, default='./model_out')
    parse.add_argument('--model_out_name', type=str, default='final_model.pth')
    parse.add_argument('--seed', type=int, default=5, help='random seed')
    parse.add_argument('--eval_model_path', type=str,
                       default='/home/tiger/projects/rscup2019_classifier/model_out/logistic_out_6.pth')
    return parse.parse_args()


def evalute(args):
    val_set = RSDataset(rootpth=args.data_dir, mode='val')
    val_loader = DataLoader(val_set,
                            batch_size=args.test_batch_size,
                            drop_last=True,
                            shuffle=True,
                            pin_memory=True,
                            num_workers=args.num_workers)
    net = Resnet34()
    net.eval()
    net.load_state_dict(torch.load(args.eval_model_path))
    net.cuda()

    total = [0 for i in range(45)]
    correct = [0 for i in range(45)]
    with torch.no_grad():
        for img, lb in val_loader:
            img, lb = img.cuda(), lb.cuda()
            outputs = net(img)
            outputs = torch.sigmoid(outputs)
            predicted = torch.max(outputs, dim=1)[1]
            res = predicted == lb

            for label_idx in range(args.test_batch_size):
                label_single = lb[label_idx]
                correct[label_single] += res[label_idx].item()
                total[label_single] += 1
            # print(correct, total)

        acc_str = 'Accuracy: {}\n'.format(sum(correct)/sum(total))
        for acc_idx in range(45):
            try:
                acc = correct[acc_idx] / total[acc_idx]
            except:
                acc = 0
            finally:
                acc_str += 'classID: {},\taccuracy: {}\n'.format(acc_idx+1, acc)
    print(acc_str)


if __name__ == '__main__':
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    evalute(args)
