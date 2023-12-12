from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from tempfile import TemporaryDirectory
import os
import os.path as osp
import numpy as np
from PIL import Image
import gc

from config import Config


class MicrogramsDataset():
# class for microscope photos

    def __init__(self):
        self.root_dir = Config['root_path']
        self.train_dir = Config['train_path']
        self.test_dir = Config['test_path']
        self.transforms = self.get_data_transforms()
        self.X_train_dir = Config['train_dataset_imgs']
        self.y_train_dir = Config['train_dataset_masks']
        # print(self.X_train_dir)


    def get_data_transforms(self):
        data_transforms = {
            'train': transforms.Compose([
                transforms.CenterCrop((370,640)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]),
            'val': transforms.Compose([
                transforms.CenterCrop((370,640)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]),
        }
        return data_transforms


    def create_train_set(self):
        X, y = [], []
        X_files = sorted([os.path.join(self.X_train_dir, file) for file in os.listdir(self.X_train_dir)])
        y_files = sorted([os.path.join(self.y_train_dir, file) for file in os.listdir(self.y_train_dir)])

        for x_item, y_item in zip(X_files, y_files):
            X.append(x_item)
            y.append(y_item)
        
        # free space
        del X_files, y_files
        gc.collect()

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=True)
        print('X_train:', np.shape(X_train))
        print('X_val:', np.shape(X_val))
        print('y_train:', np.shape(y_train))
        print('y_val:', np.shape(y_val))
        return X_train, X_val, y_train, y_val


class kitti_train(Dataset):
    def __init__(self, X_train, y_train, transform):
        self.X_train = X_train
        self.y_train = y_train
        self.transform = transform

    def __len__(self):
        return len(self.X_train)

    def __getitem__(self, item):
        return self.transform(Image.open(self.X_train[item])), self.transform(Image.open(self.y_train[item]))
   
        
        
class MicroscopyDataset(Dataset):
    def __init__(self, img_dir, cropping, channels=3, transform=None, apply_median=False):
        self.cropping_object = cropping
        self.img_dir = img_dir
        self.transform = transform
        self.crop_nums = cropping.crop_nums
        self.img_filenames = os.listdir(img_dir)
        self.apply_median = apply_median
        if channels == 1:
            self.channels = 0
        else:
            self.channels = channels

    def __len__(self):
        return (len(self.img_filenames) * self.crop_nums)

    def __getitem__(self, crop_idx):
        image_idx = crop_idx // self.crop_nums
        img_crop_idx = crop_idx % self.crop_nums
        # print(f'img index {image_idx}, img_crop_idx {img_crop_idx}')
        name = os.path.split(self.img_filenames[image_idx])[-1].split('.')[0]

        image = cv2.imread(os.path.join(self.img_dir, self.img_filenames[image_idx]), self.channels)

        crop = self.cropping_object.crop_image(image, img_crop_idx)

        if self.apply_median:
            crop = apply_median(crop, 10)
        

        if self.transform:
            crop = self.transform(crop)

        return crop, name

        



