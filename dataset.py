import torch
import torch.utils.data as data
import torchvision.transforms.functional as TF

from PIL.Image import BILINEAR

# import matplotlib
# matplotlib.use('qt5agg')
# import matplotlib.pyplot as plt
# plt.ion()


class TrainDataset(data.Dataset):

    def __init__(self, input1, input2, mask, label, length, crop_size):
        super(TrainDataset, self).__init__()

        self.in1 = input1
        self.in2 = input2
        self.label = label
        self.mask = mask
        self.length = length
        self.crop = crop_size

    def spatial_transform(self, in1, in2, mask, label):

        vflip = torch.rand(1).item() > 0.5
        hflip = torch.rand(1).item() > 0.5
        deg = torch.LongTensor(1).random_(-20, 20).item()
        scale = torch.FloatTensor(1).uniform_(0.8, 1.2)
        # x_crop = torch.randint(low=self.crop // 2, high=in1.shape[0] - self.crop // 2, size=(1,))
        # y_crop = torch.randint(low=self.crop // 2, high=in1.shape[1] - self.crop // 2, size=(1,))

        # source = list(torch.unbind(source))

        image_list = [in1, in2, mask, label]
        for i, p in enumerate(image_list):
            dtype = p.dtype

            p_min = p.min()
            p_max = p.max()

            p = (p - p_min) / ((p_max - p_min) + 0.001)

            p = TF.to_pil_image(p.float())
            if vflip:
                p = TF.vflip(p)
            if hflip:
                p = TF.hflip(p)

            p = TF.affine(p, deg, scale=scale, translate=(0, 0), shear=0, resample=BILINEAR)
            p = TF.to_tensor(p).squeeze()
            # p = p[x_crop - self.crop // 2:x_crop + self.crop // 2, y_crop - self.crop // 2:y_crop + self.crop // 2]
            if dtype == torch.int64:
                p = p.round()
            p = p.to(dtype=dtype)

            p = (p * ((p_max - p_min) + 0.001)) + p_min

            image_list[i] = p.clone()

        in1, in2, mask, label = image_list

        return in1, in2, mask, label

    @staticmethod
    def color_transform(in1, in2):

        image_list = [in1, in2]

        for i, p in enumerate(image_list):
            br = torch.FloatTensor(1).uniform_(-0.01, 0.01) + 1
            cn = torch.FloatTensor(1).uniform_(-0.01, 0.01) + 1
            sn = torch.FloatTensor(1).uniform_(-0.01, 0.01) + 1

            p_min = p.min()
            p_max = p.max()

            p = (p - p_min) / ((p_max - p_min) + 0.001)

            p = TF.to_pil_image(p.clone())
            p = TF.adjust_brightness(p, br)
            p = TF.adjust_contrast(p, cn)
            p = TF.adjust_saturation(p, sn)

            p = TF.to_tensor(p).clone()
            p = (p * ((p_max - p_min) + 0.001)) + p_min

            image_list[i] = p.clone()

        in1, in2 = image_list

        return in1, in2

    def __getitem__(self, item):

        sl = item % self.label.shape[-1]

        in1 = self.in1[:, :, sl].squeeze()
        in2 = self.in2[:, :, sl].squeeze()
        label = self.label[:, :, sl].squeeze()
        mask = self.mask[:, :, sl].squeeze()

        in1, in2, mask, label = self.spatial_transform(in1, in2, mask, label)
        in1, in2 = self.color_transform(in1, in2)

        label = (label + 1000.0) / 4000.0
        mask = mask >= 0.5

        input = torch.stack((in1.squeeze(), in2.squeeze()), dim=0)

        return input.float(), mask.bool(), label.float()

    def __len__(self):
        return self.length


class EvalDataset(data.Dataset):
    def __init__(self, input1, input2, mask, label, length):
        super(EvalDataset, self).__init__()

        self.in1 = input1
        self.in2 = input2
        self.label = label
        self.mask = mask
        self.length = length
        # self.crop = crop_size

    def __getitem__(self, item):

        in1 = self.in1[:, :, item].squeeze()
        in2 = self.in2[:, :, item].squeeze()
        label = self.label[:, :, item].squeeze()
        mask = self.mask[:, :, item].squeeze()

        label = (label + 1000.0) / 4000.0
        mask = mask >= 0.5

        input = torch.stack((in1.squeeze(), in2.squeeze()), dim=0)

        return input.float(), mask.bool(), label.float()

    def __len__(self):
        return self.length()
