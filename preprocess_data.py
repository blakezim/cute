import torch
import glob

from scipy.io import loadmat


def process_data():
    data_path = './Data/RawData/'
    out_path = './Data/PreProcessedData/'
    files = sorted(glob.glob(f'{data_path}/*'))

    train_input = []
    train_masks = []
    train_label = []

    for skull in files[:-1]:
        print(f'Processing {skull.split("/")[-1].split("_")[0]} ... ', end='')
        mat_dict = loadmat(skull)

        ute1 = torch.tensor(mat_dict['Ims1reg'])
        ute2 = torch.tensor(mat_dict['Ims2reg'])
        ct = torch.tensor(mat_dict['imsCTreg'])
        ct_mask = torch.tensor(mat_dict['CTbinaryMaskReg'])

        train_input.append(torch.stack((ute1, ute2), 0))
        train_masks.append(ct_mask)
        train_label.append(ct)

        print('done')

    train_input = torch.cat(train_input, -1)
    train_masks = torch.cat(train_masks, -1)
    train_label = torch.cat(train_label, -1)

    print(f'Processing {files[-1].split("/")[-1].split("_")[0]} ... ', end='')
    mat_dict = loadmat(files[-1])

    ute1 = torch.tensor(mat_dict['Ims1reg'])
    ute2 = torch.tensor(mat_dict['Ims2reg'])
    infer_label = torch.tensor(mat_dict['imsCTreg'])
    infer_masks = torch.tensor(mat_dict['CTbinaryMaskReg'])
    infer_input = torch.stack((ute1, ute2), 0)

    print('done')

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
