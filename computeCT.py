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
import SimpleITK as sitk
import argparse
import numpy as np
# import SimpleITK as sitk
# import torch.nn.functional as F

# import matplotlib
from models import unet_model, vnet_model

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

parser = argparse.ArgumentParser(description='PyTorch Patch Based Super Deep Interpolation Example')

parser.add_argument('-d', '--data_dir', type=str, default='./Data/PreProcessedData/skull005_as_test/',
                    help='Path to data directory')
parser.add_argument('-o', '--out_dir', type=str, default='./Output/', help='Path to output')
parser.add_argument('--trainBatchSize', type=int, default=64, help='training batch size')
parser.add_argument('--inferBatchSize', type=int, default=64, help='cross validation batch size')
parser.add_argument('--nEpochs', type=int, default=1000, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--seed', type=int, default=358, help='random seed to use. Default=358')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
# parser.add_argument('--cube_size', type=int, default=[128, 128, 128], help='3D cube size')
# parser.add_argument('--eval_size', type=int, default=[128, 128, 128], help='3D cube size')
parser.add_argument('-s', '--samples', type=int, default=200, help='how many samples from each volume')

opt = parser.parse_args()
print(opt)


def _get_branch(opt):
    p = sp.Popen(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], shell=False, stdout=sp.PIPE)
    branch, _ = p.communicate()
    branch = branch.decode('utf-8').split()

    p = sp.Popen(['git', 'rev-parse', '--short', 'HEAD'], shell=False, stdout=sp.PIPE)
    hash, _ = p.communicate()
    hash = hash.decode('utf-8').split()

    opt.git_branch = branch
    opt.git_hash = hash


def _check_branch(opt, saved_dict, model=None):
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

            device = torch.device('cuda')
            model_gpu = model.to(device=device)
            model_gpu = nn.DataParallel(model_gpu)
            params = torch.load(f'{opt.model_dir}{opt.ckpt}')
            model_gpu.load_state_dict(params['state_dict'])

            device = torch.device('cpu')
            model = vnet_model.VNet(2, 1)
            # params = torch.load(f'{opt.model_dir}{opt.ckpt}', map_location=device)
            model.load_state_dict(model_gpu.module.state_dict())
    except:
        raise Exception(f'The checkpoint {opt.ckpt} could not be loaded into the given model.')
    #
    # if saved_dict.git_branch != opt.git_branch or saved_dict.git_hash != opt.git_hash:
    #     msg = 'The model loaded, but you are not on the same branch or commit.'
    #     msg += 'To check out the right branch, run the following in the repository: \n'
    # msg += f'git checkout {params.git_branch[0]}\n'
    # msg += f'git revert {params.git_hash[0]}'
    # raise Warning(msg)

    return model, device


def add_figure(tensor, writer, title=None, text=None, text_start=[160, 200], text_spacing=40,
               label=None, cmap='jet', epoch=0, min_max=None):
    import matplotlib.pyplot as plt

    font = {'color': 'white', 'size': 20}
    plt.figure()
    if min_max:
        plt.imshow(tensor.squeeze().cpu(), cmap=cmap, vmin=min_max[0], vmax=min_max[1])
    else:
        plt.imshow(tensor.squeeze().cpu(), cmap=cmap)
    plt.colorbar()
    plt.axis('off')
    if title:
        plt.title(title)
    if text:
        if type(text) == list:
            for i, t in enumerate(text, 1):
                plt.text(text_start[0], (i * text_spacing) + text_start[1], t, fontdict=font)

    writer.add_figure(label, plt.gcf(), epoch)
    plt.close('all')


def load_infer_data(opt):
    # load the training data
    inf_files = sorted([x for x in glob.glob(f'{opt.dataDirectory}/*') if 'infer' in x])

    infer_input = torch.load([x for x in inf_files if 'input' in x][0])
    infer_label = torch.load([x for x in inf_files if 'label' in x][0])
    infer_masks = torch.load([x for x in inf_files if 'masks' in x][0])

    return infer_input, infer_masks, infer_label


def load_train_data(opt):
    # load the inference data
    train_input = torch.load(f'{opt.dataDirectory}/train_input.pt')
    train_label = torch.load(f'{opt.dataDirectory}/train_label.pt')
    train_masks = torch.load(f'{opt.dataDirectory}/train_masks.pt')

    return train_input, train_masks, train_label


def load_test_data(opt):
    # load the testing data
    train_input = torch.load(f'{opt.dataDirectory}/skull{opt.skull}_test_input.pt')
    train_label = torch.load(f'{opt.dataDirectory}/skull{opt.skull}_test_label.pt')
    train_masks = torch.load(f'{opt.dataDirectory}/skull{opt.skull}_test_masks.pt')

    return train_input, train_masks, train_label


def get_loaders(opt):
    utes, mask, label = load_train_data(opt)
    samps = opt.samples
    train_length = int(label.shape[0]) * samps

    train_dataset = TrainDataset(utes, mask, label, train_length, opt)
    train_sampler = SubsetRandomSampler(range(0, train_length))
    train_loader = DataLoader(train_dataset, opt.trainBatchSize, sampler=train_sampler, num_workers=opt.threads)

    utes, mask, label = load_infer_data(opt)

    z_dims = utes.shape[-1] * utes.shape[0]

    infer_dataset = EvalDataset(utes, mask, label, z_dims)
    infer_sampler = SequentialSampler(range(0, z_dims - 2))
    infer_loader = DataLoader(infer_dataset, opt.inferBatchSize, sampler=infer_sampler, num_workers=opt.threads)

    return train_loader, infer_loader


def WriteDICOM(grid, folder):

    dim = len(grid.size)

    # Need to put the vector in the last dimension
    vector_grid = grid.data.permute(list(range(1, dim +1)) + [0]).squeeze()  # it will always be this size now

    if dim == 2 and vector_grid.shape[-1] == 3:
        itk_image = sitk.GetImageFromArray(vector_grid.cpu().numpy(), isVector=True)
    # elif dim == 2:
    #     itk_image = sitk.GetImageFromArray(vector_grid.unsqueeze(-2).cpu().numpy())
    else:
        itk_image = sitk.GetImageFromArray(vector_grid.cpu().numpy())

    spacing = grid.spacing.tolist()
    if dim == 2:
        spacing = [1.0] + spacing

    origin = grid.origin.tolist()
    if dim == 2:
        origin = [1.0] + origin

    # ITK ordering is x, y, z. But numpy is z, y, x
    itk_image.SetSpacing(spacing[::-1])
    itk_image.SetOrigin(origin[::-1])

    if not os.path.exists(f'{folder}'):
        os.makedirs(f'{folder}')

    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()

    modification_time = time.strftime("%H%M%S")
    modification_date = time.strftime("%Y%m%d")

    direction = itk_image.GetDirection()
    series_tag_values = [("0008|0060", 'CT'),
                         ("0020|000e", "1.2.826.0.1.3680043.2.1125." + modification_date + ".1" + modification_time),
                         ("0020|0037", '\\'.join(map(str, (direction[0], direction[3], direction[6],
                                                           direction[1], direction[4], direction[7]))))]

    castFilter = sitk.CastImageFilter()
    castFilter.SetOutputPixelType(sitk.sitkInt16)
    for i in range(itk_image.GetDepth()):
        image_slice = itk_image[:, :, i]
        image_slice = castFilter.Execute(image_slice)
        # Tags shared by the series.
        for tag, value in series_tag_values:
            image_slice.SetMetaData(tag, value)
        # Slice specific tags.
        # image_slice.SetMetaData("0020|000d", '1.2.826.0.1.3680043.2.1125.1.66897271260407186792894744070373029')
        # image_slice.SetMetaData("0020|000e", '1.2.826.0.1.3680043.2.1125.1.91514226971117938304888529649661939')
        image_slice.SetMetaData("0008|0012", time.strftime("%Y%m%d"))  # Instance Creation Date
        image_slice.SetMetaData("0008|0013", time.strftime("%H%M%S"))  # Instance Creation Time
        image_slice.SetMetaData("0020|0032", '\\'.join(
            map(str, itk_image.TransformIndexToPhysicalPoint((0, 0, i)))))  # Image Position (Patient)
        # image_slice.SetMetaData("0020,0013", str(i))  # Instance Number
        # Write to the output directory and add the extension dcm, to force writing in DICOM format.
        writer.SetFileName(f'{folder}/{i:03d}.dcm')
        writer.Execute(image_slice)


def learn(opt):
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    plt.ion()

    def checkpoint(state, opt, epoch):
        path = f'{opt.outDirectory}/saves/{opt.timestr}/epoch_{epoch:05d}_model.pth'
        torch.save(state, path)
        print(f"===> Checkpoint saved for Epoch {epoch}")

    def train(epoch, scheduler):
        model.train()
        n_samps = []
        b_losses = []
        # crit = nn.MSELoss(reduction='none')
        crit = nn.MSELoss()

        for iteration, batch in enumerate(training_data_loader, 1):
            inputs, mask, label = batch[0].to(device=device), batch[1].to(device=device), batch[2].to(device=device)

            n_samps.append(mask.sum().item())
            optimizer.zero_grad()
            pred = model(inputs).squeeze()

            loss = crit(pred[mask], label[mask])
            loss.backward()

            b_loss = loss.item()
            b_losses.append(loss.item())
            optimizer.step()

            if iteration == len(training_data_loader) // 2 and epoch % 10 == 0:
                with torch.no_grad():
                    l1Loss = nn.MSELoss()
                    im = len(inputs) // 2

                    mask_slice = mask[im, :, :]
                    label_slice = label[im, :, :] * mask_slice
                    pred_slice = pred[im, :, :] * mask_slice
                    input1_slice = inputs[im, 0, :, :] * mask_slice
                    input2_slice = inputs[im, 1, :, :] * mask_slice

                    add_figure(input1_slice, writer, title='Input 1', label='Train/Input1', cmap='viridis', epoch=epoch)
                    add_figure(input2_slice, writer, title='Input 2', label='Train/Input2', cmap='viridis', epoch=epoch)

                    # Add the prediction
                    pred_loss = l1Loss(pred_slice[mask_slice], label_slice[mask_slice])
                    add_figure(pred_slice, writer, title='Predicted CT', label='Train/Pred CT', cmap='plasma',
                               epoch=epoch,
                               text=[f'Loss: {pred_loss.item():.4f}',
                                     f'Mean: {pred_slice[mask_slice].mean():.2f}',
                                     f'Min:  {pred_slice[mask_slice].min():.2f}',
                                     f'Max:  {pred_slice[mask_slice].max():.2f}'
                                     ], min_max=[0.0, 1.0], text_start=[5, 5], text_spacing=40)
                    # Add the stir
                    add_figure(label_slice, writer, title='Real CT', label='Train/Real CT', cmap='plasma', epoch=epoch,
                               text=[f'Mean: {label_slice[mask_slice].mean():.2f}',
                                     f'Min:  {label_slice[mask_slice].min():.2f}',
                                     f'Max:  {label_slice[mask_slice].max():.2f}'
                                     ], min_max=[0.0, 1.0], text_start=[5, 5], text_spacing=40)

            for param_group in optimizer.param_groups:
                clr = param_group['lr']
            writer.add_scalar('Batch/Learning Rate', clr, (iteration + (len(training_data_loader) * (epoch - 1))))
            writer.add_scalar('Batch/Avg. MSE Loss', b_loss, (iteration + (len(training_data_loader) * (epoch - 1))))
            print("=> Done with {} / {}  Batch Loss: {:.6f}".format(iteration, len(training_data_loader), b_loss))

        e_loss = (torch.tensor(n_samps) * torch.tensor(b_losses)).sum() / torch.tensor(n_samps).sum()
        writer.add_scalar('Epoch/Avg. MSE Loss', e_loss, epoch)
        # print(f"===> Avg. Loss: {e_loss:.6f}")
        print("===> Epoch {} Complete: Avg. Loss: {:.6f}".format(epoch, e_loss))
        scheduler.step(e_loss / len(training_data_loader))

    def infer(epoch):
        # crit = nn.MSELoss(reduction='none')
        n_samps = []
        b_losses = []
        crit = nn.MSELoss()

        print('===> Evaluating Model')
        with torch.no_grad():
            model.eval()
            for iteration, batch in enumerate(testing_data_loader, 1):
                inputs, mask, label = batch[0].to(device=device), batch[1].to(device=device), batch[2].to(device=device)

                n_samps.append(mask.sum().item())
                pred = model(inputs).squeeze()
                loss = crit(pred[mask], label[mask])
                b_loss = loss.item()
                b_losses.append(loss.item())

                if iteration == len(testing_data_loader) // 2 and epoch % 10 == 0:
                    im = len(inputs) // 4

                    l1Loss = nn.MSELoss()
                    mask_slice = mask[im, :, :]
                    label_slice = label[im, :, :] * mask_slice
                    pred_slice = pred[im, :, :] * mask_slice

                    if epoch == 10:
                        # Add the input images - they are not going to change
                        input1_slice = inputs[im, 0, :, :] * mask_slice
                        input2_slice = inputs[im, 1, :, :] * mask_slice
                        add_figure(input1_slice, writer, title='Input 1', label='Infer/Input1', cmap='viridis',
                                   epoch=epoch)
                        add_figure(input2_slice, writer, title='Input 2', label='Infer/Input2', cmap='viridis',
                                   epoch=epoch)
                        add_figure(label_slice, writer, title='Real CT', label='Infer/Real CT', cmap='plasma',
                                   epoch=epoch,
                                   text=[f'Mean: {label_slice[mask_slice].mean():.2f}',
                                         f'Min:  {label_slice[mask_slice].min():.2f}',
                                         f'Max:  {label_slice[mask_slice].max():.2f}'
                                         ], min_max=[0.0, 1.0])

                    # Add the prediction

                    pred_loss = l1Loss(pred_slice[mask_slice], label_slice[mask_slice])
                    add_figure(pred_slice, writer, title='Predicted CT', label='Infer/Pred T1', cmap='plasma',
                               epoch=epoch,
                               text=[f'Loss: {pred_loss.item():.4f}',
                                     f'Mean: {pred_slice[mask_slice].mean():.2f}',
                                     f'Min:  {pred_slice[mask_slice].min():.2f}',
                                     f'Max:  {pred_slice[mask_slice].max():.2f}'
                                     ], min_max=[0.0, 1.0])
                print(f"=> Done with {iteration} / {len(testing_data_loader)}  Batch Loss: {b_loss:.6f}")

            e_loss = (torch.tensor(n_samps) * torch.tensor(b_losses)).sum() / torch.tensor(n_samps).sum()
            writer.add_scalar('Infer/Avg. MSE Loss', e_loss, epoch)
            print(f"===> Avg. Loss: {e_loss:.6f}")

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

    model = unet_model.UNet(6, 1)
    model = model.to(device)
    model = nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=25, verbose=True, factor=0.5,
                                                     threshold=5e-3, cooldown=75, min_lr=1e-6)

    print("===> Beginning Training")

    epochs = range(1, opt.nEpochs + 1)

    for epoch in epochs:
        print("===> Learning Rate = {}".format(optimizer.param_groups[0]['lr']))
        train(epoch, scheduler)
        if epoch % 10 == 0:
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
    import torch.nn.functional as F

    # files = sorted(glob.glob(f'{opt.dataDirectory}/*'))

    print('===> Loading Data ... ', end='')
    opt.dataDirectory = f'{opt.dataDirectory}/skull{opt.skull}_as_test/'

    inputs, mask, label = load_test_data(opt)

    inputs = inputs.squeeze().unfold(-1, 3, 1).permute((0, 4, 1, 2, 3)).contiguous()
    inputs = inputs.view(-1, 512, 512, 400).permute(3, 0, 1, 2).split(64, 0)

    masks = mask.squeeze().unfold(-1, 3, 1).permute((3, 0, 1, 2)).contiguous()
    masks = masks.permute(3, 0, 1, 2)[:, 1].split(64, 0)

    label = label.squeeze().unfold(-1, 3, 1).permute((3, 0, 1, 2)).contiguous()
    label = label.permute(3, 0, 1, 2)[:, 1]
    label = ((label + 1000) / 4000).split(64, 0)
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

    model = unet_model.UNet(6, 1)
    saved_dict = SimpleNamespace(**torch.load(f'{opt.model_dir}{opt.ckpt}'))
    model, device = _check_branch(opt, saved_dict, model)
    print('done')

    crit = nn.MSELoss()
    e_loss = 0.0
    preds = []
    input_vols = []
    label_vol = []
    mask_vol = []
    b_losses = []
    n_samps = []

    print('===> Evaluating Model')
    with torch.no_grad():
        model.eval()
        for i, f, m, l in zip(range(1, len(inputs) + 1), inputs, masks, label):
            inputs, mask, label = f.to(device=device), m.to(device=device), l.to(device=device)

            n_samps.append(mask.sum().item())
            pred = model(inputs).squeeze()
            loss = crit(pred[mask], label[mask])
            b_loss = loss.item()
            b_losses.append(loss.item())

            preds.append(pred.clone().cpu())
            input_vols.append(inputs.clone().cpu())
            label_vol.append(label.clone().cpu())
            mask_vol.append(mask.clone().cpu())

            print(f"=> Done with {i} / {len(inputs)}  Batch Loss: {b_loss:.6f}")

        e_loss = (torch.tensor(n_samps) * torch.tensor(b_losses)).sum() / torch.tensor(n_samps).sum()
        print(f"===> Avg. Loss: {e_loss:.6f}")

        # torch.backends.cudnn.enabled = False

        # model = torch.jit.trace(model, inputs[:, :, 0:128, 0:128, 0:128].clone())

        #
        # inputs = inputs.to(device=device)
        # mask = mask.to(device=device)
        # label = label.to(device=device)

        # for param in model.parameters():
        #     param.requires_grad = False

        # pred = model(inputs).squeeze()
        # preds.append(pred.clone())
        # input_vols.append(inputs.clone())
        # label_vol.append(label.clone())
        # mask_vol.append(mask.clone())
        #
        # loss = crit(pred[mask], label[mask])
        # e_loss += loss.item()
        # b_loss = loss.item()
        #
        # print(f"===> Avg. MSE Loss: {e_loss / len(infer_loader):.6f}")

    pred_vol = F.pad(torch.cat(preds, dim=0), [0, 0, 0, 0, 1, 1])
    # mask_vol = F.pad(torch.cat(mask_vol, dim=0), [0, 0, 0, 0, 1, 1])
    label_vol = F.pad(torch.cat(label_vol, dim=0), [0, 0, 0, 0, 1, 1])
    # input_vols = torch.cat(input_vols, dim=0)
    # label_vol = torch.cat(label_vol, dim=0)

    # pred_vol = pred_vol * mask_vol
    # label_vol = label_vol * mask_vol
    pred_vol = (pred_vol * 4000.0) - 1000.0
    pred_vol[pred_vol < -1000.0] = -1000
    pred_vol[pred_vol > 3000.0] = 3000.0
    # pred_vol = pred_vol.permute(1, 2, 0)

    label_vol = (label_vol * 4000.0) - 1000.0
    label_vol[label_vol < -1000.0] = -1000
    label_vol[label_vol > 3000.0] = 3000.0
    # label_vol = label_vol.permute(1, 2, 0)

    raw_file = sorted(glob.glob(f'{opt.rawDir}skull{opt.skull}*.mat'))[-1]
    raw_dict = loadmat(raw_file)
    ct_mask = torch.tensor(raw_dict['boneMask2']).permute(2, 0, 1)
    ct_mask = ct_mask >= 0.5
    ct_mask = ct_mask.to(dtype=torch.float32)

    label_vol *= ct_mask
    pred_vol *= ct_mask

    import CAMP.Core as core
    import CAMP.FileIO as io
    import CAMP.StructuredGridTools as st
    import CAMP.UnstructuredGridOperators as uo
    import CAMP.StructuredGridOperators as so

    out_pred = core.StructuredGrid(pred_vol.shape, tensor=pred_vol.unsqueeze(0))
    out_label = core.StructuredGrid(label_vol.shape, tensor=label_vol.unsqueeze(0))

    print('Saving ... ', end='')

    WriteDICOM(out_pred, f'{opt.outDirectory}/skull{opt.skull}/skull{opt.skull}_prediction/')
    WriteDICOM(out_label, f'{opt.outDirectory}/skull{opt.skull}/skull{opt.skull}_label/')

    io.SaveITKFile(out_pred, f'{opt.outDirectory}/skull{opt.skull}/prediction_volume.nii.gz')
    io.SaveITKFile(out_label, f'{opt.outDirectory}/skull{opt.skull}/label_volume.nii.gz')

    print('done')

    # fig_dir = f'./Output/figures/{opt.model_dir.split("/")[-2]}/'
    #
    # out_dict = {
    #     'pred_CT': pred_vol.cpu().numpy(),
    #     'real_CT': label_vol.cpu().numpy(),
    #     'CT_mask': ct_mask.cpu().numpy(),
    # }
    # if not os.path.exists(fig_dir):
    #     os.makedirs(fig_dir)
    #
    # savemat(f'{fig_dir}/Skull{opt.skull}_NN_Output.mat', out_dict)

    # ute1 = input_vols[:, 0, :, :].permute(1, 2, 0)
    # ute2 = input_vols[:, 1, :, :].permute(1, 2, 0)

    # downsample = True
    # if downsample:
    #     raw_file = sorted(glob.glob(f'{opt.rawDir}skull{opt.skull}*.mat'))[0]
    #     raw_dict = loadmat(raw_file)
    #     zd = raw_dict['imsCT'].shape[-1]
    #     import CAMP.Core as core
    #     pred_grid = core.StructuredGrid(pred_vol.shape, device='cuda:1', tensor=pred_vol.unsqueeze(0))
    #     pred_grid.set_size((pred_grid.shape()[1], pred_grid.shape()[2], zd))
    #     label_grid = core.StructuredGrid(pred_vol.shape, device='cuda:1', tensor=label_vol.unsqueeze(0))
    #     label_grid.set_size((label_grid.shape()[1], label_grid.shape()[2], zd))
    #     ute1_grid = core.StructuredGrid(pred_vol.shape, device='cuda:1', tensor=ute1.unsqueeze(0))
    #     ute1_grid.set_size((ute1_grid.shape()[1], ute1_grid.shape()[2], zd))
    #     ute2_grid = core.StructuredGrid(pred_vol.shape, device='cuda:1', tensor=ute2.unsqueeze(0))
    #     ute2_grid.set_size((ute2_grid.shape()[1], ute2_grid.shape()[2], zd))
    #
    #     pred_vol = pred_grid.data.squeeze().cpu()
    #     label_vol = label_grid.data.squeeze().cpu()
    #     ute1 = ute1_grid.data.squeeze().cpu()
    #     ute2 = ute2_grid.data.squeeze().cpu()

    # s = 80
    # save_fig = True
    # fig_dir = f'./Output/figures/{opt.model_dir.split("/")[-2]}/'
    #
    # if not os.path.exists(fig_dir):
    #     os.makedirs(fig_dir)

    # in1_im = ute1[:, :, s].squeeze().cpu()
    # in2_im = ute2[:, :, s].squeeze().cpu()
    # ct_im = label_vol[:, :, s].squeeze().cpu()
    # pred_im = pred_vol[:, :, s].squeeze().cpu()
    #
    # plt.figure()
    # plt.imshow(in1_im, vmin=in1_im.min(), vmax=in1_im.max(), cmap='plasma')
    # plt.axis('off')
    # plt.title('UTE 1')
    # plt.colorbar()
    # if save_fig:
    #     plt.savefig(f'{fig_dir}ute1.png', dpi=600, bbox_inches='tight', pad_inches=0,
    #                 transparaent=True, facecolor=[0, 0, 0, 0])
    #
    # plt.figure()
    # plt.imshow(in2_im, vmin=in1_im.min(), vmax=in1_im.max(), cmap='plasma')
    # plt.axis('off')
    # plt.title('UTE 2')
    # plt.colorbar()
    # if save_fig:
    #     plt.savefig(f'{fig_dir}ute2.png', dpi=600, bbox_inches='tight', pad_inches=0,
    #                 transparaent=True, facecolor=[0, 0, 0, 0])
    #
    # plt.figure()
    # plt.imshow(pred_im, vmin=-1000.0, vmax=3000.0, cmap='gray')
    # plt.axis('off')
    # plt.title('Predicted CT')
    # plt.colorbar()
    # if save_fig:
    #     plt.savefig(f'{fig_dir}pred_ct.png', dpi=600, bbox_inches='tight', pad_inches=0,
    #                 transparaent=True, facecolor=[0, 0, 0, 0])
    #
    # plt.figure()
    # plt.imshow(ct_im, vmin=-1000.0, vmax=3000.0, cmap='gray')
    # plt.axis('off')
    # plt.title('Real CT')
    # plt.colorbar()
    # if save_fig:
    #     plt.savefig(f'{fig_dir}real_ct.png', dpi=600, bbox_inches='tight', pad_inches=0,
    #                 transparaent=True, facecolor=[0, 0, 0, 0])

    # import CAMP.Core as core
    # import CAMP.FileIO as io
    # pred_vol_out = core.StructuredGrid(pred_vol.shape, device=device, tensor=pred_vol.unsqueeze(0))
    # io.SaveITKFile(pred_vol_out, )

    # # Generate the dictionary to save
    # out_dict = {
    #     'pred_CT': pred_vol.cpu().numpy(),
    #     'real_CT': label_vol.cpu().numpy(),
    #     'UTE1': ute1.cpu().numpy(),
    #     'UTE2': ute2.cpu().numpy()
    # }
    #
    # savemat(f'{fig_dir}/Skull{opt.skull}_NN_Output.mat', out_dict)


if __name__ == '__main__':
    trainOpt = {'trainBatchSize': opt.trainBatchSize,
                'inferBatchSize': opt.inferBatchSize,
                'dataDirectory': opt.data_dir,
                'outDirectory': opt.out_dir,
                'nEpochs': opt.nEpochs,
                'lr': opt.lr,
                'cuda': opt.cuda,
                'threads': opt.threads,
                'resume': False,
                'scheduler': True,
                'ckpt': None,
                'seed': opt.seed,
                'samples': opt.samples
                }

    evalOpt = {'inferBatchSize': 16,
               'skull': '005',
               'dataDirectory': '/home/sci/blakez/ucair/cute/Data/PreProcessedData/',
               'model_dir': '/home/sci/blakez/ucair/cute/Output/saves/2020-08-09-192025/',
               'rawDir': '/hdscratch/ucair/CUTE/Data/RawData2/',
               'outDirectory': './Output/predictions/',
               'cuda': True,
               'threads': 0,
               'ckpt': 'epoch_00490_model.pth'
               }

    evalOpt = SimpleNamespace(**evalOpt)
    trainOpt = SimpleNamespace(**trainOpt)

    # learn(trainOpt)
    eval(evalOpt)
    print('All Done')
