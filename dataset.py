import torch
import torch.nn as nn
import kornia.augmentation as ka
import torch.utils.data as data


class TrainDataset(data.Dataset):

    def __init__(self, utes, mask, ct, length, opt):
        super(TrainDataset, self).__init__()

        self.utes = utes
        self.label = ct
        self.mask = mask
        self.length = length
        self.num_vols = utes.shape[0]

        self.batch_size = opt.trainBatchSize

        self.spatial = nn.Sequential(
            ka.RandomAffine(45, translate=(0.1, 0.1), scale=(0.85, 1.15), shear=(0.1, 0.1),
                            same_on_batch=True),
            ka.RandomVerticalFlip(same_on_batch=True),
            ka.RandomHorizontalFlip(same_on_batch=True)
        )
        self.dim = 2
        self.counter = 0

    def __getitem__(self, item):
        # import matplotlib
        # matplotlib.use('qt5agg')
        # import matplotlib.pyplot as plt
        # plt.ion()

        if self.counter == self.batch_size:
            self.dim = torch.LongTensor(1).random_(3).item()
            self.counter = 0

        # Get which volume to get it out of
        v = item % self.num_vols
        ims_shape = self.utes[v].shape

        rand_slice = torch.LongTensor(1).random_(ims_shape[self.dim + 1] - 2) + 1

        slicer_sing = [slice(ims_shape[0]), slice(ims_shape[1]), slice(ims_shape[2]), slice(ims_shape[3])]
        slicer_mult = [slice(ims_shape[0]), slice(ims_shape[1]), slice(ims_shape[2]), slice(ims_shape[3])]
        slicer_sing[self.dim + 1] = slice(rand_slice.item(), rand_slice.item() + 1)
        slicer_mult[self.dim + 1] = slice(rand_slice.item() - 1, rand_slice.item() + 2)

        utes = self.utes[v][slicer_mult].squeeze()

        ex_dims = list(range(1, 4))
        ex_dims.remove(self.dim + 1)
        perm = [0, self.dim + 1] + ex_dims

        utes = utes.permute(perm).contiguous().view(-1, 448, 448)
        ct = self.label[v][slicer_sing[1:]].squeeze()
        mask = self.mask[v][slicer_sing[1:]].squeeze()

        params = torch.cat([utes, ct.unsqueeze(0), mask.unsqueeze(0)])
        params = self.spatial(params).squeeze()
        inputs = params[:-2]
        label = params[-2].squeeze()
        mask = params[-1].squeeze()

        label = (label + 1000) / 4000

        mask = mask >= 0.5

        self.counter += 1

        return inputs.clone(), mask.bool(), label.float()

    def __len__(self):
        return self.length


class EvalDataset(data.Dataset):
    def __init__(self, utes, mask, ct, length):
        super(EvalDataset, self).__init__()

        self.utes = utes
        self.label = ct
        self.mask = mask

        self.num_vols = utes.shape[0]

        self.length = length

    def __getitem__(self, item):

        utes = self.utes[0, :, :, :, item: item + 3].squeeze()
        utes = utes.permute([0, 3, 1, 2]).contiguous().view(-1, 448, 448)

        ct = self.label[0, :, :, item + 1].squeeze()
        mask = self.mask[0, :, :, item + 1].squeeze()

        ct = (ct + 1000) / 4000

        mask = mask >= 0.5

        return utes.clone(), mask.bool(), ct.float()

    def __len__(self):
        return self.length
