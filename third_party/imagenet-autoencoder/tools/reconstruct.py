#!/usr/bin/env python

import argparse

import matplotlib.pyplot as plt

import torch

from torchvision.transforms import transforms

import sys
sys.path.append("./")

import utils
import models.builer as builder
import dataloader

import os
from src.utils.visualization import plot_images, plot_image_pairs
from src.utils.set_random_seeds import set_seeds
from src.utils.utils import batch2img_list
from src.utils import configs

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def get_args():
    # parse the args
    print('=> parse the args ...')
    parser = argparse.ArgumentParser(description='Trainer for auto encoder')
    parser.add_argument('--arch', default='vgg16', type=str, 
                        help='backbone architechture')
    parser.add_argument('--resume', type=str)
    parser.add_argument('--val_list', type=str)              
    parser.add_argument('--input-size', default=224, type=int,)

    args = parser.parse_args()

    args.parallel = 0
    args.batch_size = 8
    args.workers = 0

    return args

def main(args):
    fig_dir = os.path.join(configs.fig_dir, "imagenet-autoencoder-pretrained")
    os.makedirs(fig_dir, exist_ok=True)

    print('=> torch version : {}'.format(torch.__version__))

    utils.init_seeds(1, cuda_deterministic=False)

    print('=> modeling the network ...')
    model = builder.BuildAutoEncoder(args)     
    total_params = sum(p.numel() for p in model.parameters())
    print('=> num of params: {} ({}M)'.format(total_params, int(total_params * 4 / (1024*1024))))

    print('=> loading pth from {} ...'.format(args.resume))
    utils.load_dict(args.resume, model)
    
    print('=> building the dataloader ...')
    train_loader = dataloader.val_loader(args)

    model.eval()
    print('=> reconstructing ...')
    with torch.no_grad():
        for i, (input, target) in enumerate(train_loader):
            
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            output = model(input)

            input = batch2img_list(input,8)
            output = batch2img_list(output,8)

            plot_image_pairs(input, output, os.path.join(fig_dir, f"reconstruction_{i}.png"))

            if i == 10:
                break


if __name__ == '__main__':

    args = get_args()

    main(args)
    # example usage of reconstruct.py

    # python tools/reconstruct.py --arch vgg16 --resume "/cluster/scratch/horatan/mars/results/099.pth" --val_list "perseverance_navcam_color" --input-size 32

