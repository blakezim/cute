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
parser.add_argument('-o', '--out_path', type=str, default='./Data/PreProcessedData/Label003Data_HR/',
                    help='Path to save data')
parser.add_argument('-s', '--skull', type=int, default=5,
                    help='Path to save data')
opt = parser.parse_args()

device = 'cuda:1'


def process_network_data(files):

    inputs = []
    masks = []
    labels = []

    for i, skull in enumerate(files):
        print(f'Processing {skull.split("/")[-1].split("_")[0]} ... ')
        mat_dict = loadmat(skull)
        skull_num = skull.split('/')[-1].split('_')[0]

        # if os.path.exists(f'{skull.split("skull")[0]}{skull_num}_registered_UTEtoCT_FlexSpine.mat'):
        #     print('Using High Res')
        #     ute_mat = loadmat(f'{skull.split("skull")[0]}{skull_num}_registered_UTEtoCT_FlexSpine.mat')
        #     utes = torch.tensor(ute_mat['imsUTEreg2']).permute([-1] + list(range(0, 3)))
        # else:
        utes = torch.tensor(mat_dict['imsUTEreg']).permute([-1] + list(range(0, 3)))

        s = utes.shape
        means = torch.mean(utes[:, s[1] // 2 - 20: s[1] // 2 + 20, s[2] // 2 - 20: s[2] // 2 + 20,
                           s[3] // 2 - 20: s[3] // 2 + 20], dim=(1, 2, 3)).view([-1] + [1] * 3)
        utes = utes / means
        ct = torch.tensor(mat_dict['imsCTreg'])
        ct_mask = torch.tensor(mat_dict['boneMask2'])

        ct_mask = torch.tensor(binary_dilation(ct_mask, iterations=2))

        nz_z = torch.LongTensor([i for i in range(0, ct.shape[-1]) if ct_mask[:, :, i].sum() != 0][1:-1])
        nz_y = torch.LongTensor([i for i in range(0, ct.shape[-2]) if ct_mask[:, i, :].sum() != 0][1:-1])
        nz_x = torch.LongTensor([i for i in range(0, ct.shape[-3]) if ct_mask[i, :, :].sum() != 0][1:-1])

        rs_z, rs_y, rs_x = 448, 448, 448

        ct_out = core.StructuredGrid(
            list(ct[:, :, nz_z][:, nz_y, :][nz_x, :, :].shape),
            tensor=ct[:, :, nz_z][:, nz_y, :][nz_x, :, :].unsqueeze(0),
            channels=1
        ).set_size((rs_x, rs_y, rs_z), inplace=False)

        mask_out = core.StructuredGrid(
            list(ct_mask[:, :, nz_z][:, nz_y, :][nz_x, :, :].shape),
            tensor=ct_mask[:, :, nz_z][:, nz_y, :][nz_x, :, :].unsqueeze(0),
            channels=1
        ).set_size((rs_x, rs_y, rs_z), inplace=False)

        utes_out = core.StructuredGrid(
            list(utes[:, :, :, nz_z][:, :, nz_y, :][:, nz_x, :, :].shape)[1:],
            tensor=utes[:, :, :, nz_z][:, :, nz_y, :][:, nz_x, :, :],
            channels=utes[:, :, :, nz_z][:, :, nz_y, :][:, nz_x, :, :].shape[0]
        ).set_size((rs_x, rs_y, rs_z), inplace=False)

        inputs.append(utes_out.data.cpu().squeeze())
        masks.append((mask_out.data.cpu().squeeze() >= 0.5).float())
        labels.append(ct_out.data.cpu().squeeze())

        print(f'Processing {skull.split("/")[-1].split("_")[0]} ... done')

    inputs = torch.stack(inputs, 0)
    masks = torch.stack(masks, 0)
    labels = torch.stack(labels, 0)

    return inputs, masks, labels


def process_test_data(files):

    inputs = []
    masks = []
    labels = []

    for i, skull in enumerate(files):
        print(f'Processing {skull.split("/")[-1].split("_")[0]} ... ')
        mat_dict = loadmat(skull)
        skull_num = skull.split('/')[-1].split('_')[0]
        # if os.path.exists(f'{skull.split("skull")[0]}{skull_num}_registered_UTEtoCT_FlexSpine.mat'):
        #     print('Using High Res')
        #     ute_mat = loadmat(f'{skull.split("skull")[0]}{skull_num}_registered_UTEtoCT_FlexSpine.mat')
        #     utes = torch.tensor(ute_mat['imsUTEreg2']).permute([-1] + list(range(0, 3)))
        # else:
        utes = torch.tensor(mat_dict['imsUTEreg']).permute([-1] + list(range(0, 3)))

        s = utes.shape
        means = torch.mean(utes[:, s[1] // 2 - 20: s[1] // 2 + 20, s[2] // 2 - 20: s[2] // 2 + 20,
                           s[3] // 2 - 20: s[3] // 2 + 20], dim=(1, 2, 3)).view([-1] + [1] * 3)
        utes = utes / means
        ct = torch.tensor(mat_dict['imsCTreg'])
        ct_mask = torch.tensor(mat_dict['boneMask2'])

        ct_mask = torch.tensor(binary_dilation(ct_mask, iterations=2))

        inputs.append(utes)
        masks.append(ct_mask)
        labels.append(ct)

        print(f'Processing {skull.split("/")[-1].split("_")[0]} ... done')

    inputs = torch.stack(inputs, 0)
    masks = torch.stack(masks, 0)
    labels = torch.stack(labels, 0)

    return inputs, masks, labels


def process_data(opt):
    data_path = opt.data_path

    files = sorted(glob.glob(f'{data_path}/*.mat'))
    files = [x for x in files if 'workspace' in x]

    out_path = f'./Data/PreProcessedData/skull{opt.skull:03d}_as_test/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # Check the save path for a infer skull
    inf_files = sorted([x for x in glob.glob(f'{out_path}/*') if 'infer' in x])

    if inf_files:
        infer_skull = f'skull{inf_files[0].split("skull")[-1].split("_")[0]}'
        infer_file = files.pop(files.index([x for x in files if infer_skull in x][0]))
    else:
        infer_file = files.pop(np.random.randint(0, len(files)))
        infer_skull = f'skull{infer_file.split("skull")[-1].split("_")[0]}'

    test_file = files.pop(files.index([x for x in files if f'skull{opt.skull:03d}' in x][0]))
    train_files = files

    train_input, train_masks, train_label = process_network_data(train_files)
    infer_input, infer_masks, infer_label = process_network_data([infer_file])
    test_input, test_masks, test_label = process_test_data([test_file])

    print('Saving ... ', end='')
    torch.save(train_input, f'{out_path}/train_input.pt')
    torch.save(train_masks, f'{out_path}/train_masks.pt')
    torch.save(train_label, f'{out_path}/train_label.pt')

    torch.save(infer_input, f'{out_path}/{infer_skull}_infer_input.pt')
    torch.save(infer_masks, f'{out_path}/{infer_skull}_infer_masks.pt')
    torch.save(infer_label, f'{out_path}/{infer_skull}_infer_label.pt')

    torch.save(test_input, f'{out_path}/skull{opt.skull:03d}_test_input.pt')
    torch.save(test_masks, f'{out_path}/skull{opt.skull:03d}_test_masks.pt')
    torch.save(test_label, f'{out_path}/skull{opt.skull:03d}_test_label.pt')

    print('done')


if __name__ == '__main__':
    process_data(opt)
