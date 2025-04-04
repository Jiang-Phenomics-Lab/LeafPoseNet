

import os
import json
import torch
import numpy as np
import pandas as pd
import cv2
import torch.utils.data as data
import random
from sklearn.model_selection import train_test_split

class LeafKeypoint(data.Dataset):
    def __init__(self,
                 #repeats = 5,
                 train="train",
                 transforms=None):
        super().__init__()
        self.flag = train 
        self.transforms = transforms

        df = pd.read_excel('datasets/labels/labels.xlsx')
        df = df.sort_values(by='img_name')
        random.seed(66)
        df = df.sample(frac=1, random_state=66).reset_index(drop=True)

        # all_varieties = list(df['id'].unique())

        # id_counts = df['id'].value_counts()

        # varieties_with_10_rows = id_counts[id_counts == 10].index.tolist()
        # test_data = df[df['id'].isin(all_varieties)].groupby('id').apply(lambda x: x.iloc[0:2]) 

        # validation_varieties = random.sample(varieties_with_10_rows, 177)  

        # validation_data = df[df['id'].isin(validation_varieties)].groupby('id').apply(lambda x: x.iloc[2:4]) 
        # train_data = df[df['id'].isin(validation_varieties)].groupby('id').apply(lambda x: x.iloc[4:10])  #
        
        train_data, temp_data = train_test_split(df, test_size=0.4, random_state=42)
        validation_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

        if train == 'train':
            data = train_data
        elif train == 'val':
            data = validation_data
        elif train == 'test':
            data = test_data

        
        self.data = data
        
        self.info = []
        obj_idx = 0
        for num in range(len(data)):
            data_r = np.array(data.iloc[num]).flatten().tolist()
            keypoint = np.zeros((3, 2))
            keypoint[0] = [data_r[5], data_r[8]]
            keypoint[1] = [data_r[6], data_r[9]]
            keypoint[2] = [data_r[7], data_r[10]]
                
            target = {}
            target['keypoints'] = keypoint.copy()
            target['ketpoint_path'] = None
            target['img_path'] = data_r[3]
            target['obj_index'] = obj_idx
            target['score'] = 1
            obj_idx += 1
            self.info.append(target)



    def random_hsv(self, img):
        # Apply random brightness adjustment
        r = np.random.uniform(-1, 1, 3) * [0.1, 0.2, 0.1] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
        dtype = img.dtype
        xsv = np.arange(0, 256, dtype=r.dtype)
        xh = np.arange(0, 180, dtype=r.dtype)
        lut_hue = np.clip(xh * r[1], 0, 179).astype(dtype)
        lut_sat = np.clip(xsv * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(xsv * r[2], 0, 255).astype(dtype)

        img = cv2.merge((lut_hue[hue], lut_sat[sat], lut_val[val]))
        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

        return img


    def __getitem__(self, idx):
        target = self.info[idx]
        image = cv2.imread(target['img_path'])[:,:,::-1]
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transforms is not None:
            if self.flag == 'train':
                image = self.random_hsv(image)
            image, person_info = self.transforms(image, target)

        return image, person_info

    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate_fn(batch):
        imgs_tuple, targets_tuple = tuple(zip(*batch))
        imgs_tensor = torch.stack(imgs_tuple)
        return imgs_tensor, targets_tuple
