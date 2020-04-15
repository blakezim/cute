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
# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')

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


def _check_branch(opt, saved_dict, model):
    """ When performing eval, check th git branch and commit that was used to generate the .pt file"""

    # Check the current branch and hash
    _get_branch(opt)

    try:
        if opt.cuda:
            device = torch.device('cuda')
            model = model.to(device=device)
            model = nn.DataParallel(model)
            params = torch.load(f'{opt.model_dir}{opt.ckpt}')
            model.load_state_dict(params['state_dict'])
        else:
            device = torch.device('cpu')
            params = torch.load(f'{opt.model_dir}{opt.ckpt}', map_location='cpu')
            model.load_state_dict(params['state_dict'])
    except:
        raise Exception(f'The checkpoint {opt.ckpt} could not be loaded into the given model.')

    if saved_dict.git_branch != opt.git_branch or saved_dict.git_hash != opt.git_hash:
        msg = 'The model loaded, but you are not on the same branch or commit.'
        msg += 'To check out the right branch, run the following in the repository: \n'
        msg += f'git checkout {params.git_branch[0]}\n'
        msg += f'git revert {params.git_hash[0]}'
        raise Warning(msg)

    return model, device


def add_figure(tensor, writer, title=None, text=None, label=None, cmap='jet', epoch=0, vmin=None, vmax=None):

    import matplotlib.pyplot as plt

    font = {'color': 'white', 'size': 20}
    plt.figure()
    plt.imshow(tensor.squeeze().cpu(), cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.axis('off')
    if title:
        plt.title(title)
    if text:
        if type(text) == list:
            for i, t in enumerate(text, 1):
                plt.text(160, (i * 40) + 200, t, fontdict=font)

    writer.add_figure(label, plt.gcf(), epoch)
    plt.close('all')


def load_infer_data(opt):

    # load the training data
    infer_input = torch.load(f'{opt.dataDirectory}/infer_input.pt')
    infer_label = torch.load(f'{opt.dataDirectory}/infer_label.pt')
    infer_masks = torch.load(f'{opt.dataDirectory}/infer_masks.pt')

    return infer_input[0].squeeze(), infer_input[1].squeeze(), infer_masks, infer_label


def load_train_data(opt):

    # load the inference data
    train_input = torch.load(f'{opt.dataDirectory}/train_input.pt')
    train_label = torch.load(f'{opt.dataDirectory}/train_label.pt')
    train_masks = torch.load(f'{opt.dataDirectory}/train_masks.pt')

    return train_input[0].squeeze(), train_input[1].squeeze(), train_masks, train_label


def get_loaders(opt):

    input1, input2, mask, label = load_train_data(opt)

    train_dataset = TrainDataset(input1, input2, mask, label, int(label.shape[-1]), opt.crop)
    train_sampler = SubsetRandomSampler(range(0, int(label.shape[-1])))
    train_loader = DataLoader(train_dataset, opt.trainBatchSize, sampler=train_sampler, num_workers=opt.threads)

    input1, input2, mask, label = load_infer_data(opt)

    infer_dataset = EvalDataset(input1, input2, mask, label, int(label.shape[-1]))
    infer_sampler = SequentialSampler(range(0, int(label.shape[-1])))
    infer_loader = DataLoader(infer_dataset, opt.inferBatchSize, sampler=infer_sampler, num_workers=opt.threads)

    return train_loader, infer_loader


def learn(opt):

    import matplotlib
    matplotlib.use('qt5agg')
    import matplotlib.pyplot as plt
    plt.ion()

    def checkpoint(state, opt, epoch):
        path = f'{opt.outDirectory}/saves/{opt.timestr}/epoch_{epoch:05d}_model.pth'
        torch.save(state, path)
        print(f"===> Checkpoint saved for Epoch {epoch}")

    def train(epoch, scheduler):
        model.train()
        # crit = nn.MSELoss(reduction='none')
        crit = nn.L1Loss()
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
        crit = nn.L1Loss()
        e_loss = 0.0

        print('===> Evaluating Model')
        with torch.no_grad():
            for iteration, batch in enumerate(testing_data_loader, 1):
                inputs, mask, label = batch[0].to(device=device), batch[1].to(device=device), batch[2].to(device=device)

                pred = model(inputs).squeeze()
                # loss = (crit(pred.squeeze(), label) * mask).mean()
                loss = crit(pred[mask], label[mask])
                e_loss += loss.item()
                b_loss = loss.item()

                if iteration == int(len(testing_data_loader) // 2):
                    im = int(len(testing_data_loader) // 2) % opt.inferBatchSize

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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=25, verbose=True, factor=0.5,
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

    import matplotlib
    matplotlib.use('qt5agg')
    import matplotlib.pyplot as plt
    plt.ion()

    files = sorted(glob.glob(f'{opt.dataDirectory}/*'))

    print('===> Loading Data ... ', end='')
    mat_dict = loadmat(files[-1])

    ute1 = torch.tensor(mat_dict['Ims1reg'])
    ute2 = torch.tensor(mat_dict['Ims2reg'])
    ct = torch.tensor(mat_dict['imsCTreg'])
    ct_mask = torch.tensor(mat_dict['UTEbinaryMaskReg'])

    infer_dataset = EvalDataset(ute1, ute2, ct_mask, ct, int(ct.shape[-1]))
    infer_sampler = SequentialSampler(range(0, int(ct.shape[-1])))
    eval_loader = DataLoader(infer_dataset, opt.evalBatchSize, sampler=infer_sampler, num_workers=opt.threads)
    print('done')

    print('===> Loading Model ... ', end='')
    if not opt.ckpt:
        timestamps = sorted(glob.glob(f'{opt.model_dir}/*'))
        if not timestamps:
            raise Exception(f'No save directories found in {opt.model_dir}')
        lasttime = timestamps[-1].split('/')[-1]
        models = sorted(glob.glob(f'{opt.model_dir}/{lasttime}/*'))
        if not models:
            raise Exception(f'No models found in the last run ({opt.model_dir}{lasttime}/')
        model_file = models[-1].split('/')[-1]
        opt.ckpt = f'{lasttime}/{model_file}'

    model = unet_model.UNet(2, 1)
    saved_dict = SimpleNamespace(**torch.load(f'{opt.model_dir}{opt.ckpt}'))
    model, device = _check_branch(opt, saved_dict, model)
    print('done')

    model.eval()
    crit = nn.MSELoss()
    e_loss = 0.0
    preds = []

    print('===> Evaluating Model')
    with torch.no_grad():
        for iteration, batch in enumerate(eval_loader, 1):
            inputs, mask, label = batch[0].to(device=device), batch[1].to(device=device), batch[2].to(device=device)

            pred = model(inputs).squeeze()
            preds.append(pred.clone())

            loss = crit(pred[mask], label[mask])
            e_loss += loss.item()
            b_loss = loss.item()

            print(f"=> Done with {iteration} / {len(eval_loader)}  Batch Loss: {b_loss:.6f}")

        print(f"===> Avg. MSE Loss: {e_loss / len(eval_loader):.6f}")

    pred_vol = torch.cat(preds, dim=0)
    pred_vol = (pred_vol * 4000.0) - 1000.0
    pred_vol[pred_vol < -1000.0] = -1000
    pred_vol[pred_vol > 3000.0] = 3000.0
    pred_vol = pred_vol.permute(1, 2, 0)

    s = 300
    save_fig = True
    fig_dir = f'./Output/figures/{opt.model_dir.split("/")[-2]}/'

    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    in1_im = ute1[:, :, s].squeeze().cpu()
    in2_im = ute2[:, :, s].squeeze().cpu()
    ct_im = ct[:, :, s].squeeze().cpu()
    pred_im = pred_vol[:, :, s].squeeze().cpu()

    plt.figure()
    plt.imshow(in1_im, vmin=in1_im.min(), vmax=in1_im.max(), cmap='plasma')
    plt.axis('off')
    plt.title('UTE 1')
    plt.colorbar()
    if save_fig:
        plt.savefig(f'{fig_dir}ute1.png', dpi=600, bbox_inches='tight', pad_inches=0,
                    transparaent=True, facecolor=[0, 0, 0, 0])

    plt.figure()
    plt.imshow(in2_im, vmin=in1_im.min(), vmax=in1_im.max(), cmap='plasma')
    plt.axis('off')
    plt.title('UTE 2')
    plt.colorbar()
    if save_fig:
        plt.savefig(f'{fig_dir}ute2.png', dpi=600, bbox_inches='tight', pad_inches=0,
                    transparaent=True, facecolor=[0, 0, 0, 0])

    plt.figure()
    plt.imshow(pred_im, vmin=-1000.0, vmax=3000.0, cmap='gray')
    plt.axis('off')
    plt.title('Predicted CT')
    plt.colorbar()
    if save_fig:
        plt.savefig(f'{fig_dir}pred_ct.png', dpi=600, bbox_inches='tight', pad_inches=0,
                    transparaent=True, facecolor=[0, 0, 0, 0])

    plt.figure()
    plt.imshow(ct_im, vmin=-1000.0, vmax=3000.0, cmap='gray')
    plt.axis('off')
    plt.title('Real CT')
    plt.colorbar()
    if save_fig:
        plt.savefig(f'{fig_dir}real_ct.png', dpi=600, bbox_inches='tight', pad_inches=0,
                    transparaent=True, facecolor=[0, 0, 0, 0])

    # import CAMP.Core as core
    # import CAMP.FileIO as io
    # pred_vol_out = core.StructuredGrid(pred_vol.shape, device=device, tensor=pred_vol.unsqueeze(0))
    # io.SaveITKFile(pred_vol_out, )

    # Generate the dictionary to save
    out_dict = {
        'pred_CT': pred_vol.cpu().numpy(),
    }

    savemat(f'{fig_dir}/NN_Output.mat', out_dict)


if __name__ == '__main__':
    trainOpt = {'trainBatchSize': 28,
                'inferBatchSize': 28,
                'dataDirectory': '../Data/PreProcessedData',
                'outDirectory': '../Output/',
                'nEpochs': 1000,
                'lr': 0.0002,
                'cuda': True,
                'threads': 20,
                'resume': False,
                'scheduler': True,
                'ckpt': None,
                'seed': 223,
                'crop': None
                }

    evalOpt = {'evalBatchSize': 16,
               'dataDirectory': './Data/RawData/',
               'model_dir': '/home/sci/blakez/ucair/CUTE/Output/saves/2020-04-11-085151/',
               'outDirectory': './Output/Predictions/',
               'cuda': True,
               'threads': 0,
               'ckpt': 'epoch_00050_model.pth'
               }

    evalOpt = SimpleNamespace(**evalOpt)
    trainOpt = SimpleNamespace(**trainOpt)

    learn(trainOpt)
    # eval(evalOpt)
    print('All Done')
