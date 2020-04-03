import os
import sys
# import glob
import time
import torch
import glob
# import losses
import torch.nn as nn
import subprocess as sp
import torch.optim as optim
import argparse
import numpy as np
# import SimpleITK as sitk
# import torch.nn.functional as F

# import matplotlib
from models import unet_model

# from VNet.vNetModel import vnet_model
# from collections import OrderedDict
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.ion()

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from scipy.io import loadmat, savemat
from types import SimpleNamespace

from dataset import TrainDataset, EvalDataset

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler


def _get_branch(opt):
    p = sp.Popen(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], shell=False, stdout=sp.PIPE)
    branch, _ = p.communicate()
    branch = branch.decode('utf-8').split()

    p = sp.Popen(['git', 'rev-parse', '--short', 'HEAD'], shell=False, stdout=sp.PIPE)
    hash, _ = p.communicate()
    hash = hash.decode('utf-8').split()

    opt.git_branch = branch
    opt.git_hash = hash


def _check_branch(opt, params):
    """ When performing eval, check th git branch and commit that was used to generate the .pt file"""

    # Check the current branch and hash
    _get_branch(opt)

    if params.git_branch != opt.git_branch or params.git_hash != opt.git_hash:
        msg = 'You are not on the right branch or commit. Please run the following in the repository: \n'
        msg += f'git checkout {params.git_branch[0]}\n'
        msg += f'git revert {params.git_hash[0]}'
        sys.exit(msg)


def add_figure(tensor, writer, title=None, text=None, label=None, cmap='jet', epoch=0, vmin=None, vmax=None):
    font = {'color': 'white', 'size': 16}
    plt.figure()
    plt.imshow(tensor.squeeze().cpu(), cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.axis('off')
    if title:
        plt.title(title)
    if text:
        if type(text) == list:
            for i, t in enumerate(text, 1):
                plt.text(3, i*10, t, fontdict=font)

    writer.add_figure(label, plt.gcf(), epoch)
    plt.close('all')


def load_infer_data():

    # load the training data
    data_path = './Data/PreProcessedData/'

    infer_input = torch.load(f'{data_path}/infer_input.pt')
    infer_label = torch.load(f'{data_path}/infer_label.pt')
    infer_masks = torch.load(f'{data_path}/infer_masks.pt')

    return infer_input[0].squeeze(), infer_input[1].squeeze(), infer_masks, infer_label


def load_train_data():
    # load the inference data
    data_path = './Data/PreProcessedData/'

    train_input = torch.load(f'{data_path}/train_input.pt')
    train_label = torch.load(f'{data_path}/train_label.pt')
    train_masks = torch.load(f'{data_path}/train_masks.pt')

    return train_input[0].squeeze(), train_input[1].squeeze(), train_masks, train_label


def get_loaders(opt):

    input1, input2, mask, label = load_train_data()

    train_dataset = TrainDataset(input1, input2, mask, label, int(label.shape[-1]), opt.crop)
    train_sampler = SubsetRandomSampler(range(0, int(label.shape[-1])))
    train_loader = DataLoader(train_dataset, opt.trainBatchSize, sampler=train_sampler, num_workers=opt.threads)

    input1, input2, mask, label = load_train_data()

    infer_dataset = EvalDataset(input1, input2, mask, label, int(label.shape[-1]))
    infer_sampler = SequentialSampler(range(0, int(label.shape[-1])))
    infer_loader = DataLoader(infer_dataset, opt.inferBatchSize, sampler=infer_sampler, num_workers=opt.threads)

    return train_loader, infer_loader


def learn(opt):
    def checkpoint(state, opt, epoch):
        path = f'{opt.outDirectory}/saves/{opt.timestr}/epoch_{epoch:05d}_model.pth'
        torch.save(state, path)
        print(f"===> Checkpoint saved for Epoch {epoch}")

    def train(epoch, scheduler):
        model.train()
        # crit = nn.MSELoss(reduction='none')
        crit = nn.MSELoss()
        e_loss = 0.0

        for iteration, batch in enumerate(training_data_loader, 1):
            inputs, mask, label = batch[0].to(device=device), batch[1].to(device=device), batch[2].to(device=device)

            optimizer.zero_grad()
            pred = model(inputs).squeeze()

            # loss = (crit(pred.squeeze(), label) * mask).mean()
            loss = crit(pred[mask], label[mask])
            loss.backward()

            e_loss += loss.item()
            b_loss = loss.item()
            optimizer.step()

            for param_group in optimizer.param_groups:
                clr = param_group['lr']
            writer.add_scalar('Batch/Learning Rate', clr, (iteration + (len(training_data_loader) * (epoch - 1))))
            writer.add_scalar('Batch/Avg. MSE Loss', b_loss, (iteration + (len(training_data_loader) * (epoch - 1))))
            print("=> Done with {} / {}  Batch Loss: {:.6f}".format(iteration, len(training_data_loader), b_loss))

        writer.add_scalar('Epoch/Avg. MSE Loss', e_loss / len(training_data_loader), epoch)
        print("===> Epoch {} Complete: Avg. Loss: {:.6f}".format(epoch, e_loss / len(training_data_loader)))
        scheduler.step(e_loss / len(training_data_loader))

    def infer(epoch):
        model.eval()
        # crit = nn.MSELoss(reduction='none')
        crit = nn.MSELoss()
        e_loss = 0.0

        print('===> Evaluating Model')
        with torch.no_grad():
            for iteration, batch in enumerate(testing_data_loader, 1):
                inputs, mask, label = batch[0].to(device=device), batch[1].to(device=device, dtype=torch.bool), \
                                  batch[2].to(device=device)

                pred = model(inputs).squeeze()
                # loss = (crit(pred.squeeze(), label) * mask).mean()
                loss = crit(pred[mask], label[mask])
                e_loss += loss.item()
                b_loss = loss.item()

                if iteration == int(len(testing_data_loader // 2)) // opt.inferBatchSize:
                    im = int(len(testing_data_loader // 2)) % opt.inferBatchSize

                    if epoch == 1:
                        # Add the input images - they are not going to change
                        input1 = inputs[im, 0].squeeze()
                        input2 = inputs[im, 1].squeeze()
                        add_figure(input1, writer, title='Input 1', label='Infer/Input1', cmap='viridis', epoch=epoch,
                                   text=[f'Mean: {input1.mean():.2f}',
                                         f'Min:  {input1.min():.2f}',
                                         f'Max:  {input1.max():.2f}'],
                                   )
                        add_figure(input2, writer, title='Input 2', label='Infer/Input2', cmap='viridis', epoch=epoch,
                                   text=[f'Mean: {input2.mean():.2f}',
                                         f'Min:  {input2.min():.2f}',
                                         f'Max:  {input2.max():.2f}'],
                                   )
                    # Add the prediction
                    pred_im = pred[im].squeeze()
                    add_figure(pred_im, writer, title='Prediction', label='Infer/Pred', cmap='gray', epoch=epoch,
                               text=[f'Mean: {pred_im.mean():.2f}',
                                     f'Min:  {pred_im.min():.2f}',
                                     f'Max:  {pred_im.max():.2f}'],
                               vmin=0.0, vmax=1.0
                               )

                    # Add the CT - not going to change
                    if epoch == 1:
                        stir_im = label[im]

                        add_figure(stir_im, writer, title='CT', label='Infer/CT', cmap='gray', epoch=epoch,
                                   text=[f'Mean: {stir_im.mean():.2f}',
                                         f'Min:  {stir_im.min():.2f}',
                                         f'Max:  {stir_im.max():.2f}'],
                                   vmin=0.0, vmax=1.0
                                   )
                    print(f"=> Done with {iteration} / {len(testing_data_loader)}  Batch Loss: {b_loss:.6f}")

            writer.add_scalar('Infer/Avg. MSE Loss', e_loss / len(testing_data_loader), epoch)
            print(f"===> Avg. MSE Loss: {e_loss / len(testing_data_loader):.6f}")

    # Add the git information to the opt
    _get_branch(opt)
    timestr = time.strftime("%Y-%m-%d-%H%M%S")
    opt.timestr = timestr
    writer = SummaryWriter(f'{opt.outDirectory}/runs/{timestr}')
    writer.add_text('Parameters', opt.__str__())

    # Seed anything random to be generated
    torch.manual_seed(opt.seed)

    try:
        os.stat(f'{opt.outDirectory}/saves/{timestr}/')
    except OSError:
        os.makedirs(f'{opt.outDirectory}/saves/{timestr}/')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('===> Generating Datasets ... ', end='')
    training_data_loader, testing_data_loader = get_loaders(opt)
    print(' done')

    model = unet_model.UNet(2, 1)
    model = model.to(device)
    model = nn.DataParallel(model)

    optimizer = optim.SGD(model.parameters(), lr=opt.lr, weight_decay=1e-5, momentum=0.8, nesterov=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, verbose=True, factor=0.5,
                                                     threshold=5e-3, cooldown=200, min_lr=1e-6)

    print("===> Beginning Training")

    epochs = range(1, opt.nEpochs + 1)

    for epoch in epochs:
        print("===> Learning Rate = {}".format(optimizer.param_groups[0]['lr']))
        train(epoch, scheduler)
        if epoch % 50 == 0:
            checkpoint({
                'epoch': epoch,
                'scheduler': opt.scheduler,
                'git_branch': opt.git_branch,
                'git_hash': opt.git_hash,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()}, opt, epoch)
        infer(epoch)


def eval(opt):
    pass


if __name__ == '__main__':
    trainOpt = {'trainBatchSize': 16,
                'inferBatchSize': 16,
                'dataDirectory': './Data/',
                'outDirectory': './Output/',
                'nEpochs': 1000,
                'lr': 0.0001,
                'cuda': True,
                'threads': 20,
                'resume': False,
                'scheduler': True,
                'ckpt': None,
                'seed': 223,
                'crop': None
                }

    evalOpt = {'evalBatchSize': 1024,
               'dataDirectory': './Data/',
               'model_dir': './Output/saves/',
               'outDirectory': './Output/Predictions/',
               'cuda': True,
               'threads': 0,
               'ckpt': None
               }

    evalOpt = SimpleNamespace(**evalOpt)
    trainOpt = SimpleNamespace(**trainOpt)

    learn(trainOpt)
    # eval(evalOpt)
    print('All Done')
