#!/usr/bin/env python

import argparse

import matplotlib.pyplot as plt

import torch

from torchvision.transforms import transforms
from tqdm import tqdm
import sys

sys.path.append("./")

import utils
import models.builer as builder
import dataloader

import os
from src.utils.visualize_latent_space import image_scatter, knn, pca, lle, tSNE, single_cluster, latent_interpolation, mean_shift
from src.utils.set_random_seeds import set_seeds
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
    parser.add_argument('--input-size', default=32, type=int,)

    args = parser.parse_args()

    args.parallel = 0
    args.batch_size = 512
    args.workers = 0

    return args


def main(args):
    print('=> torch version : {}'.format(torch.__version__))

    utils.init_seeds(1, cuda_deterministic=False)

    print('=> modeling the network ...')
    model = builder.BuildAutoEncoder(args)
    total_params = sum(p.numel() for p in model.parameters())
    print('=> num of params: {} ({}M)'.format(total_params, int(total_params * 4 / (1024 * 1024))))

    print('=> loading pth from {} ...'.format(args.resume))
    utils.load_dict(args.resume, model)

    print('=> building the dataloader ...')
    train_loader = dataloader.val_loader(args)

    plt.figure(figsize=(16, 9))

    latent_features = torch.zeros((len(train_loader.dataset), 512))
    dataset = train_loader.dataset

    fig_dir = os.path.join(configs.fig_dir, "imagenet_autoencoder")

    model.eval()
    print('=> reconstructing ...')
    with torch.no_grad():
        for i, (input, target) in enumerate(tqdm(train_loader)):

            input = input.cuda(non_blocking=True)
            output = model.module.encoder(input)
            endix = min((i+1)*args.batch_size, len(dataset))
            startix = i*args.batch_size
            bs = endix - startix

            latent_features[startix:endix] = output.cpu().view(bs, latent_features.shape[1])


        # tSNE(latent_features, dataset, fig_dir)
        pca(latent_features)
        single_cluster(latent_features, dataset, 30, figure_dir=os.path.join(fig_dir, "kmeans_cluster"))
        # mean_shift(latent_features, dataset, figure_dir=os.path.join(fig_dir, "mean_shift_cluster"))
        # knn(latent_features, dataset)
        image_scatter(latent_features[:, 2:], dataset, range(len(dataset)),
                      filename=os.path.join(fig_dir, "latent_space.png"))



if __name__ == '__main__':
    args = get_args()

    main(args)

    # usage
    # python visualize_clustering.py --arch vgg16 --resume "/cluster/scratch/horatan/mars/results/099.pth" --val_list "perseverance_navcam_color" --input-size 32

