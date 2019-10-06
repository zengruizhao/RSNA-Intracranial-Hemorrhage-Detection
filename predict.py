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

from rs_dataset import RSDataset_test
from get_logger import get_logger
from res_network import Resnet18, Resnet34, Resnet101
from tqdm import tqdm


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--epoch', type=int, default=15)
    parse.add_argument('--schedule_step', type=int, default=2)

    parse.add_argument('--test_batch_size', type=int, default=128)
    parse.add_argument('--num_workers', type=int, default=16)

    parse.add_argument('--eval_fre', type=int, default=1)
    parse.add_argument('--msg_fre', type=int, default=10)
    parse.add_argument('--save_fre', type=int, default=2)

    parse.add_argument('--name', type=str, default='res34_baseline',
                       help='unique out file name of this task include log/model_out/tensorboard log')
    parse.add_argument('--data_dir', type=str, default='/media/tiger/zzr/rsna')
    parse.add_argument('--log_dir', type=str, default='./logs')
    parse.add_argument('--tensorboard_dir', type=str, default='./tensorboard')
    parse.add_argument('--model_out_dir', type=str, default='./model_out')
    parse.add_argument('--model_out_name', type=str, default='final_model.pth')
    parse.add_argument('--seed', type=int, default=5, help='random seed')
    parse.add_argument('--eval_model_path', type=str, default='/media/tiger/zzr/rsna_script/model_out/191005-002929_temp/out_9.pth')
    return parse.parse_args()


def predict(args):
    test_set = RSDataset_test(rootpth=args.data_dir, mode='test')
    test_loader = DataLoader(test_set,
                            batch_size=args.test_batch_size,
                            drop_last=False,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=args.num_workers)
    net = Resnet18()
    net.eval()
    net.load_state_dict(torch.load(args.eval_model_path))
    net.cuda()

    labels = ["epidural", "intraparenchymal", "intraventricular",
              "subarachnoid", "subdural", "any"]
    with open("result.csv", 'w') as fp:
        fp.write('ID,Label\n')
        with torch.no_grad():
            for img, name in tqdm(test_loader):
                img = img.cuda()
                outputs = net(img).cpu().numpy()
                for idx, i in enumerate(name):
                    for idxj, j in enumerate(labels):
                        fp.write(i + '_' + j + ',' + str(outputs[idx][idxj]) + '\n')


if __name__ == '__main__':
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    predict(args)
