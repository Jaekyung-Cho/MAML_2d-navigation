"""
MAML training all
 * Copyright (C) 2021 {Jaekyung Cho} <{jackyoung96@snu.ac.kr}>
 * 
 * This file is part of MAML-RL implementation project, ARIL, SNU
 * 
 * permission of Jaekyung Cho
 
"""

import torch
import torch.nn as nn
from torch.nn.modules.linear import Linear
import yaml
import numpy as np

from tqdm import trange


from maml import maml
from baseline1 import pretrained
from baseline2 import random_init
from baseline3 import oracle
            
 


if __name__ =='__main__':
    import argparse
    
    parser = argparse.ArgumentParser("MAML Training")
    parser.add_argument('--config', type=str, default="configs/2d-navigation.yaml")
    # parser.add_argument('--policy', type=str, required=True)
    parser.add_argument('--gpu', type=int, default=3)
    parser.add_argument('--render', action="store_true")

    misc = parser.add_argument_group('Miscellaneous')
    misc.add_argument('--use-cuda', action='store_true')
    misc.add_argument('--pretrained', action='store_true')
    misc.add_argument('--test', action='store_true')

    exct = parser.add_argument_group("Execution")
    exct.add_argument('--maml', action='store_true')
    exct.add_argument('--bs1', action='store_true')
    exct.add_argument('--bs2', action='store_true')
    exct.add_argument('--bs3', action='store_true')

    args = parser.parse_args()

    args.device = ('cuda:%d'%args.gpu if (torch.cuda.is_available()
                   and args.use_cuda) else 'cpu')

    if args.maml:
        maml(args)

    if args.bs1:
        pretrained(args)
    
    if args.bs2:
        random_init(args)
    
    if args.bs3:
        oracle(args)