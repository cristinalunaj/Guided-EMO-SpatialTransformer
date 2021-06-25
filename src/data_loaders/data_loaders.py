import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class Plain_Dataset(Dataset):
    def __init__(self, csv_path, dataroot, transform, sep=";"):
        '''
        Pytorch Dataset class
                 image, labels
                 :param csv_path: the path of the csv file    (train, validation, test)
                 :param dataroot: the directory of the images (train, validation, test)
                 :param transform: pytorch transformation over the data
                 :param sep: Column separator of the dataframe.
        '''
        self.df_file = pd.read_csv(csv_path, sep=sep, header=0)
        self.labels = self.df_file['emotion']
        self.img_dir = self.df_file['path']
        self.dataroot = dataroot
        self.transform = transform


    def __len__(self):
        return len(self.df_file)

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img = Image.open(os.path.join(self.dataroot, self.img_dir[idx].split('.')[0] + '.png')).convert('L')
        labels = np.array(self.labels[idx])
        labels = torch.from_numpy(labels).long()

        if self.transform :
            img = self.transform(img)

        return img, labels, idx


