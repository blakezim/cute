import os
import torch
import glob
import argparse
import numpy as np
import torch.optim as optim

from scipy.io import loadmat

import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
plt.ion()

from scipy.ndimage.morphology import *
import CAMP.Core as core
import CAMP.FileIO as io
import CAMP.StructuredGridTools as st
import CAMP.UnstructuredGridOperators as uo
import CAMP.StructuredGridOperators as so


parser = argparse.ArgumentParser(description='Multi-Flip Ange T1 Prediction Data Pre-Processing')

parser.add_argument('-d', '--data_path', type=str, default='/hdscratch/ucair/CUTE/Data/RawData2/',
                    help='Raw Data Path')
parser.add_argument('-o', '--out_path', type=str, default='./Data/PreProcessedData/Label005Data/',
                    help='Path to save data')
opt = parser.parse_args()

device = 'cuda:1'


def process_training_data(files):

    train_input = []
    train_masks = []
    train_label = []

    for i, skull in enumerate(files):
        print(f'Processing {skull.split("/")[-1].split("_")[0]} ... ')
        mat_dict = loadmat(skull)

        utes = torch.tensor(mat_dict['imsUTEreg']).permute([-1] + list(range(0, 3)))
        ct = torch.tensor(mat_dict['imsCTreg'])
        ct_mask = torch.tensor(mat_dict['boneMask2'])

        ct_mask = torch.tensor(binary_dilation(ct_mask, iterations=15))

        train_input.append(utes)
        train_masks.append(ct_mask)
        train_label.append(ct)

        print(f'Processing {skull.split("/")[-1].split("_")[0]} ... done')

    train_input = torch.stack(train_input, 0)
    train_masks = torch.stack(train_masks, 0)
    train_label = torch.stack(train_label, 0)

    # nz_mask = train_masks.max(dim=0)[0].max(dim=0)[0].to(torch.bool)
    #
    # train_input = train_input[:, :, :, nz_mask]
    # train_masks = train_masks[:, :, nz_mask]
    # train_label = train_label[:, :, nz_mask]

    return train_input, train_masks, train_label


def process_label_data(files):

    label_input = []
    label_masks = []
    label_label = []

    for i, skull in enumerate(files):
        print(f'Processing {skull.split("/")[-1].split("_")[0]} ... ')
        mat_dict = loadmat(skull)

        utes = torch.tensor(mat_dict['imsUTEreg']).permute([-1] + list(range(0, 3)))
        ct = torch.tensor(mat_dict['imsCTreg'])
        ct_mask = torch.tensor(mat_dict['boneMask2'])

        ct_mask = torch.tensor(binary_dilation(ct_mask, iterations=15))

        label_input.append(utes)
        label_masks.append(ct_mask)
        label_label.append(ct)

        print(f'Processing {skull.split("/")[-1].split("_")[0]} ... done')

    label_input = torch.stack(label_input, 0)
    label_masks = torch.stack(label_masks, 0)
    label_label = torch.stack(label_label, 0)

    # nz_mask = label_masks.max(dim=0)[0].max(dim=0)[0].to(torch.bool)
    #
    # label_input = label_input[:, :, :, nz_mask]
    # label_masks = label_masks[:, :, nz_mask]
    # label_label = label_label[:, :, nz_mask]

    return label_input, label_masks, label_label


def process_data(opt):
    data_path = opt.data_path
    out_path = opt.out_path

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    files = sorted(glob.glob(f'{data_path}/*.mat'))

    train_input, train_masks, train_label = process_training_data([files[0], files[1], files[2], files[3]])
    infer_input, infer_masks, infer_label = process_label_data([files[4]])

    # print(f'Processing {files[-1].split("/")[-1].split("_")[0]} ... ')
    # mat_dict = loadmat(files[-1])
    #
    # ct_mask = torch.tensor(mat_dict['CTbinaryMaskReg'])
    #
    # ute1 = torch.tensor(mat_dict['Ims1reg'])
    # ute2 = torch.tensor(mat_dict['Ims2reg'])
    # infer_label = torch.tensor(mat_dict['imsCTreg'])
    # infer_masks = torch.tensor(binary_dilation(ct_mask.numpy(), iterations=4))
    # infer_input = torch.stack((ute1, ute2), 0)
    #
    # nz_mask = infer_masks.max(dim=0)[0].max(dim=0)[0].to(torch.bool)
    #
    # infer_label = infer_label[:, :, nz_mask]
    # infer_masks = infer_masks[:, :, nz_mask]
    # infer_input = infer_input[:, :, :, nz_mask]
    #
    # print(f'Processing {files[-1].split("/")[-1].split("_")[0]} ... done')

    print('Saving ... ', end='')
    torch.save(train_input, f'{out_path}/train_input.pt')
    torch.save(train_masks, f'{out_path}/train_masks.pt')
    torch.save(train_label, f'{out_path}/train_label.pt')

    torch.save(infer_input, f'{out_path}/infer_input.pt')
    torch.save(infer_masks, f'{out_path}/infer_masks.pt')
    torch.save(infer_label, f'{out_path}/infer_label.pt')

    print('done')


if __name__ == '__main__':
    process_data(opt)
