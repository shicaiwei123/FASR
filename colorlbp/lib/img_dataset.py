from lib.processing_utils import FaceDection
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as tt
from lib.processing_utils import get_file_list, get_mean_std
import cv2
from PIL import Image
import os
import numpy as np


class ImgBinaryDataset(Dataset):

    def __init__(self, living_dir, spoofing_dir, balance=True, data_transform=None):

        self.living_path_list = get_file_list(living_dir)
        self.spoofing_path_list = get_file_list(spoofing_dir)
        self.face_detector = FaceDection(model_name='TF')

        # 间隔取样,控制数量
        if balance:
            self.spoofing_path_list = sorted(self.spoofing_path_list)
            balance_factor = int(np.floor(len(self.spoofing_path_list) / len(self.living_path_list)))
            self.spoofing_path_list = self.spoofing_path_list[0:len(self.spoofing_path_list):balance_factor]
        self.img_path_list = self.spoofing_path_list + self.living_path_list
        self.data_transform = data_transform

    def __getitem__(self, idx):
        img_path = self.img_path_list(idx)
        img = cv2.imread(img_path)
        face_img = self.face_detector.face_detect(img)

        if self.data_transform is not None:
            face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face_img_pil = Image.fromarray(face_img_rgb)
            face_img_trans = self.data_transform(face_img_pil)
            face_img = face_img_trans

        # 确定label
        img_path_split = img_path.split('/')
        img_type = img_path_split[-2]
        if img_type == 'spoofing':
            label = 0
        elif img_type == 'living':
            label = 1
        else:
            print("路径错误")
            label = 0

        return face_img, label

    def __len__(self):
        return len(self.img_path_list)


class deeppix_dataset(Dataset):

    def __init__(self, living_dir, spoofing_dir, args, data_transform=None, sampe_interal=1):

        self.living_path_list = get_file_list(living_dir)
        self.spoofing_path_list = get_file_list(spoofing_dir)

        # 间隔取样,控制数量
        if args.balance:
            self.spoofing_path_list = sorted(self.spoofing_path_list)
            balance_factor = int(np.round(len(self.spoofing_path_list) / len(self.living_path_list)))
            if balance_factor < 1:
                balance_factor = 1
            self.spoofing_path_list = self.spoofing_path_list[0:len(self.spoofing_path_list):balance_factor]

        self.spoofing_path_list = self.spoofing_path_list[0:len(self.spoofing_path_list):sampe_interal]
        self.living_path_list = self.living_path_list[0:len(self.living_path_list):sampe_interal]

        self.img_path_list = self.spoofing_path_list + self.living_path_list
        self.data_transform = data_transform
        self.face_detector = FaceDection("TF")
        self.args = args

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, idx):

        img_path = self.img_path_list[idx]
        img_path_split = img_path.split('/')

        # label
        if img_path_split[-2] == 'spoofing':

            spoofing_label = 0
            map_x = np.zeros((14, 14))

        else:
            spoofing_label = 1
            map_x = np.ones((14, 14))

        image_x = cv2.imread(img_path)
        # face_img = self.face_detector.face_detect(image_x)
        face_img = image_x
        face_img_resize = cv2.resize(face_img, (224, 224))

        # 格式转换
        face_img_rgb = cv2.cvtColor(face_img_resize, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_img_rgb)

        if self.data_transform:

            image_pil = self.data_transform(face_pil)
            image_arr = np.array(image_pil)

            map_x = torch.from_numpy(map_x)

            spoofing_label = np.array(spoofing_label, np.long)
            spoofing_label = torch.from_numpy(spoofing_label)

            sample = {'image_x': image_arr, 'map_x': map_x, 'spoofing_label': spoofing_label}
        else:
            sample = {'image_x': image_x, 'map_x': map_x, 'spoofing_label': spoofing_label}
        return sample
