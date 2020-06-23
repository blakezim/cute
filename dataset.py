import torch
import torch.utils.data as data
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from PIL.Image import BILINEAR


import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
plt.ion()


class TrainDataset(data.Dataset):

    def __init__(self, input1, input2, mask, label, length, cube_size):
        super(TrainDataset, self).__init__()

        self.in1 = input1
        self.in2 = input2
        self.label = label
        self.mask = mask
        self.length = length
        self.cube = cube_size
        self.num_vols = input1.shape[0]

    def __getitem__(self, item):
        v = item % self.num_vols

        in1 = self.in1[v, :, :, :].squeeze()
        in2 = self.in2[v, :, :, :].squeeze()
        label = self.label[v, :, :, :].squeeze()
        mask = self.mask[v, :, :, :].squeeze().float()

        sub_mask = torch.zeros(self.cube)

        label = (label + 1000.0) / 4000.0
        mask = mask >= 0.5

        while sub_mask.sum() == 0.0:
            start_x = (torch.rand(1) * (in1.shape[0] - self.cube[0])).int()
            start_y = (torch.rand(1) * (in1.shape[1] - self.cube[1])).int()
            start_z = (torch.rand(1) * (in1.shape[2] - self.cube[2])).int()

            sub_mask = mask[start_x:start_x + self.cube[0],
                       start_y:start_y + self.cube[1],
                       start_z:start_z + self.cube[2]]

        sub_in1 = in1[start_x:start_x + self.cube[0],
                  start_y:start_y + self.cube[1],
                  start_z:start_z + self.cube[2]]
        sub_in2 = in2[start_x:start_x + self.cube[0],
                  start_y:start_y + self.cube[1],
                  start_z:start_z + self.cube[2]]

        sub_label = label[start_x:start_x + self.cube[0],
                    start_y:start_y + self.cube[1],
                    start_z:start_z + self.cube[2]]

        input = torch.stack((sub_in1.squeeze(), sub_in2.squeeze()), dim=0)

        return input.float(), sub_mask.bool(), sub_label.float()

    def __len__(self):
        return self.length


class EvalDataset(data.Dataset):
    def __init__(self, input1, input2, mask, label):
        super(EvalDataset, self).__init__()

        self.in1 = input1.squeeze()[64:-64, 64:-64, 9:-9].contiguous()
        self.in2 = input2.squeeze()[64:-64, 64:-64, 9:-9].contiguous()
        self.label = label.squeeze()[64:-64, 64:-64, 9:-9].contiguous()
        self.mask = mask.squeeze()[64:-64, 64:-64, 9:-9].contiguous()

        self.in1 = self.block_vol(self.in1)
        self.in2 = self.block_vol(self.in2)
        self.label = self.block_vol(self.label)
        self.mask = self.block_vol(self.mask)
        self.length = len(self.in1)

    @staticmethod
    def block_vol(vol):
        import matplotlib
        matplotlib.use('qt5agg')
        import matplotlib.pyplot as plt
        plt.ion()
        vol = vol.unfold(2, 128, 128).unfold(1, 128, 128).unfold(0, 128, 128).contiguous()
        vol = vol.view(-1, 128, 128, 128).contiguous()

        return vol

    def __getitem__(self, item):
        in1 = self.in1[item].squeeze()
        in2 = self.in2[item].squeeze()
        label = self.label[item].squeeze()
        mask = self.mask[item].squeeze()

        label = (label + 1000.0) / 4000.0
        mask = mask >= 0.5

        input = torch.stack((in1.squeeze(), in2.squeeze()), dim=0)

        return input.float(), mask.bool(), label.float()

    def __len__(self):
        return self.length


class PredictDataset(data.Dataset):
    def __init__(self, input1, input2, mask, label):
        super(PredictDataset, self).__init__()

        # self.in1 = input1.squeeze()[64:-64, 64:-64, 9:-9].contiguous()
        # self.in2 = input2.squeeze()[64:-64, 64:-64, 9:-9].contiguous()
        # self.label = label.squeeze()[64:-64, 64:-64, 9:-9].contiguous()
        # self.mask = mask.squeeze()[64:-64, 64:-64, 9:-9].contiguous()

        # test = F.pad(input1, (23, 23, 0, 0, 0, 0))
        pad = 128 - (torch.tensor(input1.squeeze().shape) % 128)
        pad[pad == 128] = 0
        pad_array = [0] * len(pad) * 2
        pad_array[-2] = pad[-1].item()
        pad_array = pad_array[::-1]

        self.in1 = self.block_vol(F.pad(input1.squeeze(), pad_array))
        self.in2 = self.block_vol(F.pad(input2.squeeze(), pad_array))
        self.label = self.block_vol(F.pad(label.squeeze(), pad_array))
        self.mask = self.block_vol(F.pad(mask.squeeze(), pad_array))
        self.length = len(self.in1)

    @staticmethod
    def block_vol(vol):
        # vol = F.pad(vol, [8, 8, 8, 8, 8, 8])
        vol = vol.unfold(2, 128, 64).unfold(1, 128, 64).unfold(0, 128, 64).contiguous()
        vol = vol.view(-1, 128, 128, 128).contiguous()


        return vol

    def __getitem__(self, item):
        in1 = self.in1[item].squeeze()
        in2 = self.in2[item].squeeze()
        label = self.label[item].squeeze()
        mask = self.mask[item].squeeze()

        label = (label + 1000.0) / 4000.0
        mask = mask >= 0.5

        input = torch.stack((in1.squeeze(), in2.squeeze()), dim=0)

        return input.float(), mask.bool(), label.float()

    def __len__(self):
        return self.length
