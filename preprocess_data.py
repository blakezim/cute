import torch
import glob
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

device = 'cuda:1'


def affine_register(target, source, converge=1.0, niter=300, device='cpu', rigid=True):

    gaussian_blur = so.Gaussian.Create(1, 25, 15, dim=3, device=device)
    target = gaussian_blur(target)
    source = gaussian_blur(source)

    # Do some additional registration just to make sure it is in the right spot
    similarity = so.L2Similarity.Create(device=device, dim=3)
    model = so.AffineIntensity.Create(similarity, dim=3, device=device)

    # Create the optimizer
    optimizer = optim.SGD([
        {'params': model.affine, 'lr': 1.0e-10},
        {'params': model.translation, 'lr': 1.0e-10}], momentum=0.9, nesterov=True
    )

    energy = []
    for epoch in range(0, niter):
        optimizer.zero_grad()
        loss = model(
            target.data, source.data
        )
        energy.append(loss.item())

        print(f'===> Iteration {epoch:3} Energy: {loss.item():.3f}')

        loss.backward()  # Compute the gradients
        optimizer.step()  #

        if rigid:
            with torch.no_grad():
                U, s, V = model.affine.clone().svd()
                model.affine.data = torch.mm(U, V.transpose(1, 0))

        # if epoch >= 2:
        if epoch > 10 and np.mean(energy[-10:]) - energy[-1] < converge:
            break

    itr_affine = torch.eye(4, device=device, dtype=torch.float32)
    itr_affine[0:3, 0:3] = model.affine
    itr_affine[0:3, 3] = model.translation

    return itr_affine


def deformable_register(target, source, converge=1.0, niter=300, device='cpu'):

    gaussian_blur = so.Gaussian.Create(1, 25, 10, dim=3, device=device)
    target = gaussian_blur(target)
    source = gaussian_blur(source)
    #
    # source = (source - source.min()) / (source.max() - source.min())
    # target = (target - target.min()) / (target.max() - target.min())

    deformation = target.clone()
    deformation.set_to_identity_lut_()
    deformation_list = []

    # Create a grid composer
    composer = so.ComposeGrids(device=device, dtype=torch.float32, padding_mode='border')

    steps = [0.03, 0.05]

    for i, s in enumerate([4, 2]):

        scale_source = source.set_size(source.size // s, inplace=False)
        scale_target = target.set_size(target.size // s, inplace=False)
        deformation = deformation.set_size(target.size // s, inplace=False)

        # Apply the deformation to the source image
        scale_source = so.ApplyGrid(deformation)(scale_source)

        # Create the matching term
        similarity = so.L2Similarity.Create(dim=3, device=device)

        # Create the smoothing operator
        operator = so.FluidKernel.Create(
            scale_target,
            device=device,
            alpha=1.0,
            beta=0.0,
            gamma=0.001,
        )

        # Now register the source and the gad volume
        interday = st.IterativeMatch.Create(
            source=scale_source,
            target=scale_target,
            similarity=similarity,
            operator=operator,
            device=device,
            step_size=steps[i],
            incompressible=True
        )

        energy = [interday.initial_energy]
        print(f'Iteration: 0   Energy: {interday.initial_energy}')
        for i in range(1, niter):
            energy.append(interday.step())
            print(f'Iteration: {i}   Energy: {energy[-1]}')

            if i > 10 and np.mean(energy[-10:]) - energy[-1] < converge:
                break

        deformation = interday.get_field()
        deformation_list.append(deformation.clone().set_size(source.size, inplace=False))
        deformation = composer(deformation_list[::-1])

    return deformation


def process_data():
    data_path = './Data/RawData/'
    out_path = './Data/PreProcessedData/'
    files = sorted(glob.glob(f'{data_path}/*.mat'))

    train_input = []
    train_masks = []
    train_label = []

    for i, skull in enumerate(files[:-1]):
        print(f'Processing {skull.split("/")[-1].split("_")[0]} ... ')
        mat_dict = loadmat(skull)

        ute1 = torch.tensor(mat_dict['Ims1reg'])
        ute2 = torch.tensor(mat_dict['Ims2reg'])
        ct = torch.tensor(mat_dict['imsCTreg'])
        # ute_mask = torch.tensor(mat_dict['UTEbinaryMaskReg'])
        ct_mask = torch.tensor(mat_dict['CTbinaryMaskReg'])

        # if i == 1:
        #     source = core.StructuredGrid(ct_mask.shape, device=device, tensor=ct_mask.unsqueeze(0))
        #     target = core.StructuredGrid(ute_mask.shape, device=device, tensor=ute_mask.unsqueeze(0))
        #
        #     affine = affine_register(target, source, device=device, niter=100, converge=10.0)
        #     app_def = so.AffineTransform.Create(affine, device=device)
        #     def_ct = app_def(core.StructuredGrid(ct_mask.shape, device=device, tensor=ct.unsqueeze(0)))
        #     io.SaveITKFile(def_ct, f'/home/sci/blakez/ct_skull{i}_def.nii.gz')

        # start_vol = ct_mask.sum() / 1000

        # ute1 = core.StructuredGrid(ute1.shape, device=device, tensor=ute1.unsqueeze(0))
        # ute2 = core.StructuredGrid(ute2.shape, device=device, tensor=ute2.unsqueeze(0))
        # ct = core.StructuredGrid(ct.shape, device=device, tensor=ct.unsqueeze(0))
        # ct_mask = core.StructuredGrid(ct_mask.shape, device=device, tensor=ct_mask.unsqueeze(0))

        # io.SaveITKFile(ute1, f'/home/sci/blakez/ute1_skull{i}.nii.gz')
        # io.SaveITKFile(ute2, f'/home/sci/blakez/ute2_skull{i}.nii.gz')
        # io.SaveITKFile(ct, f'/home/sci/blakez/ct_skull{i}.nii.gz')
        # io.SaveITKFile(ct_mask, f'/home/sci/blakez/ct_mask_skull{i}.nii.gz')


        #
        # deformation = deformable_register(target, source, device=device, niter=100, converge=10.0)
        #
        # app_def = so.ApplyGrid.Create(deformation, device=device)
        # def_ct = app_def(core.StructuredGrid(ct_mask.shape, device=device, tensor=ct.unsqueeze(0)))
        # app_def = so.ApplyGrid.Create(deformation, interp_mode='nearest', device=device)
        # def_mask = app_def(core.StructuredGrid(ct_mask.shape, device=device, tensor=ct_mask.unsqueeze(0)))
        #
        # ct_mask = def_mask.data.squeeze().cpu().numpy()
        # print(f'Starting Volume = {start_vol}')
        # print(f'Ending Volume = {np.sum(ct_mask) / 1000}')
        ct_mask = torch.tensor(binary_dilation(ct_mask.numpy(), iterations=10))

        train_input.append(torch.stack((ute1, ute2), 0))
        train_masks.append(ct_mask)
        train_label.append(ct)
        # train_label.append(def_ct.data.squeeze().cpu())

        print(f'Processing {skull.split("/")[-1].split("_")[0]} ... done')

    train_input = torch.cat(train_input, -1)
    train_masks = torch.cat(train_masks, -1)
    train_label = torch.cat(train_label, -1)

    nz_mask = train_masks.max(dim=0)[0].max(dim=0)[0].to(torch.bool)

    train_input = train_input[:, :, :, nz_mask]
    train_masks = train_masks[:, :, nz_mask]
    train_label = train_label[:, :, nz_mask]

    print(f'Processing {files[-1].split("/")[-1].split("_")[0]} ... ')
    mat_dict = loadmat(files[-1])

    # ute1 = torch.tensor(mat_dict['Ims1reg'])
    # ute2 = torch.tensor(mat_dict['Ims2reg'])
    # ct = torch.tensor(mat_dict['imsCTreg'])
    # ute_mask = torch.tensor(mat_dict['UTEbinaryMaskReg'])
    ct_mask = torch.tensor(mat_dict['CTbinaryMaskReg'])

    # ute1 = core.StructuredGrid(ute1.shape, device=device, tensor=ute1.unsqueeze(0))
    # ute2 = core.StructuredGrid(ute2.shape, device=device, tensor=ute2.unsqueeze(0))
    # ct = core.StructuredGrid(ct.shape, device=device, tensor=ct.unsqueeze(0))
    # ct_mask = core.StructuredGrid(ct_mask.shape, device=device, tensor=ct_mask.unsqueeze(0))
    #
    # io.SaveITKFile(ute1, '/home/sci/blakez/ute1_skull2.nii.gz')
    # io.SaveITKFile(ute2, '/home/sci/blakez/ute2_skull2.nii.gz')
    # io.SaveITKFile(ct, '/home/sci/blakez/ct_skull2.nii.gz')
    # io.SaveITKFile(ct_mask, '/home/sci/blakez/ct_mask_skull2.nii.gz')

    # start_vol = ct_mask.sum() / 1000
    #
    # source = core.StructuredGrid(ct_mask.shape, device=device, tensor=ct_mask.unsqueeze(0))
    # target = core.StructuredGrid(ute_mask.shape, device=device, tensor=ute_mask.unsqueeze(0))
    #
    # deformation = deformable_register(target, source, device=device, niter=100, converge=10.0)
    #
    # app_def = so.ApplyGrid.Create(deformation, device=device)
    # def_ct = app_def(core.StructuredGrid(ct_mask.shape, device=device, tensor=ct.unsqueeze(0)))
    # app_def = so.ApplyGrid.Create(deformation, interp_mode='nearest', device=device)
    # def_mask = app_def(core.StructuredGrid(ct_mask.shape, device=device, tensor=ct_mask.unsqueeze(0)))
    #
    # ct_mask = def_mask.data.squeeze().cpu().numpy()
    # print(f'Starting Volume = {start_vol}')
    # print(f'Ending Volume = {np.sum(ct_mask) / 1000}')
    # ct_mask = torch.tensor(binary_dilation(ct_mask, iterations=10))

    # infer_input = torch.stack((ute1, ute2), 0)
    # infer_masks = ct_mask
    # infer_label = ct
    # infer_label = def_ct.data.squeeze().cpu()

    ute1 = torch.tensor(mat_dict['Ims1reg'])
    ute2 = torch.tensor(mat_dict['Ims2reg'])
    infer_label = torch.tensor(mat_dict['imsCTreg'])
    infer_masks = torch.tensor(binary_dilation(ct_mask.numpy(), iterations=10))
    infer_input = torch.stack((ute1, ute2), 0)

    nz_mask = infer_masks.max(dim=0)[0].max(dim=0)[0].to(torch.bool)

    infer_label = infer_label[:, :, nz_mask]
    infer_masks = infer_masks[:, :, nz_mask]
    infer_input = infer_input[:, :, :, nz_mask]

    print(f'Processing {files[-1].split("/")[-1].split("_")[0]} ... done')

    print('Saving ... ', end='')
    torch.save(train_input, f'{out_path}/train_input.pt')
    torch.save(train_masks, f'{out_path}/train_masks.pt')
    torch.save(train_label, f'{out_path}/train_label.pt')

    torch.save(infer_input, f'{out_path}/infer_input.pt')
    torch.save(infer_masks, f'{out_path}/infer_masks.pt')
    torch.save(infer_label, f'{out_path}/infer_label.pt')

    print('done')


if __name__ == '__main__':
    process_data()
