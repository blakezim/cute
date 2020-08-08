import torch
import torch.utils.data as data
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from PIL.Image import BILINEAR


# import matplotlib
# matplotlib.use('qt5agg')
# import matplotlib.pyplot as plt
# plt.ion()


class TrainDataset(data.Dataset):

    def __init__(self, utes, mask, ct, length, opt):
        super(TrainDataset, self).__init__()

        self.utes = utes
        self.label = ct
        self.mask = mask
        self.length = length
        self.num_vols = utes.shape[0]

        self.batch_size = opt.trainBatchSize

        self.dim = 2
        self.counter = 0

    def spatial_transform(self, ute, mask, ct):

        vflip = torch.rand(1).item() > 0.5
        hflip = torch.rand(1).item() > 0.5
        deg = torch.LongTensor(1).random_(-10, 10).item()
        scale = torch.FloatTensor(1).uniform_(0.8, 1.2)
        translate = tuple(torch.FloatTensor(2).uniform_(0.0, 0.10).tolist())
        shear = torch.FloatTensor(1).uniform_(0.0, 3.0).item()

        image_list = [ute, mask, ct]
        for i, p in enumerate(image_list):

            if len(p.shape) > 2:
                for s in range(p.shape[0]):
                    sl = p[s]
                    dtype = sl.dtype

                    sl_min = sl.min()
                    sl_max = sl.max()

                    sl = (sl - sl_min) / ((sl_max - sl_min) + 0.001)

                    sl = TF.to_pil_image(sl.float())
                    if vflip:
                        sl = TF.vflip(sl)
                    if hflip:
                        sl = TF.hflip(sl)

                    sl = TF.affine(sl, deg, scale=scale, translate=translate, shear=shear, resample=BILINEAR)
                    sl = TF.to_tensor(sl).squeeze()
                    if dtype == torch.int64:
                        sl = sl.round()
                    sl = sl.to(dtype=dtype)

                    sl = (sl * ((sl_max - sl_min) + 0.001)) + sl_min
                    p[s] = sl
                noise = torch.FloatTensor(p.size()).normal_(0, 0.05)
                p = (p + noise).clone()

            else:
                p_min = p.min()
                p_max = p.max()

                p = (p - p_min) / ((p_max - p_min) + 0.001)

                p = TF.to_pil_image(p.float())
                if vflip:
                    p = TF.vflip(p)
                if hflip:
                    p = TF.hflip(p)

                p = TF.affine(p, deg, scale=scale, translate=translate, shear=shear, resample=BILINEAR)
                p = TF.to_tensor(p).squeeze()
                if dtype == torch.int64:
                    p = p.round()
                p = p.to(dtype=dtype)

                p = (p * ((p_max - p_min) + 0.001)) + p_min

            image_list[i] = p.clone()
        ute, mask, ct = image_list

        return ute, mask, ct

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

        # Spatially trasform the source and target
        utes, mask, ct = self.spatial_transform(utes, mask, ct)

        ct = (ct + 1000) / 4000

        mask = mask >= 0.5

        self.counter += 1

        return utes.clone(), mask.bool(), ct.float()

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
