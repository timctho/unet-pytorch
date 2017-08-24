import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
import os
import numpy as np
from torchvision.transforms import Compose



class CarvanaDataset(Dataset):
    def __init__(self, csv_path, img_dir, mask_dir, transform=None):
        super(CarvanaDataset, self).__init__()
        self.dataset_csv = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataset_csv)

    def __getitem__(self, idx):
        # Image path
        img_name = self.dataset_csv.ix[idx, 0]
        img_path = os.path.join(self.img_dir, img_name)

        # Mask path
        mask_path = os.path.join(self.mask_dir, img_name.split('.')[0] + '_mask.png')


        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, 0)


        img = cv2.resize(img, (1024, 1024))
        mask = cv2.resize(mask, (1024, 1024))

        mask = np.expand_dims(mask, axis=2)
        img = np.transpose(img, [2, 0, 1]).astype(np.float32)
        mask = np.transpose(mask, [2, 0, 1]).astype(np.float32)


        mask /= 255.0
        img -= 128.0


        data_pair = {'img': torch.from_numpy(img), 'mask': torch.from_numpy(mask)}
        return data_pair


if __name__ == '__main__':

    dataset = CarvanaDataset('train_masks.csv', 'train', 'train_masks_png')

    data_loader = DataLoader(dataset, 30, True, num_workers=4)

    for idx, batch_data in enumerate(data_loader):
        print(idx)
        print(batch_data['img'].size())
        print(batch_data['mask'].size())

        for i in range(batch_data['img'].size()[0]):
            img = batch_data['img'][i]
            mask = batch_data['mask'][i]
            print(type(img), type(mask))

            img = img.numpy()
            mask = mask.numpy()

            img_r = cv2.resize(img, (256, 256))
            mask_r = cv2.resize(mask, (256, 256))
            mask_r = np.expand_dims(mask_r, axis=2)
            mask_r = np.repeat(mask_r, 3, 2)
            concat_img = np.concatenate((img_r, mask_r), axis=1)

            cv2.imshow('m', concat_img)
            cv2.waitKey(0)