# Load the datase

import torch.utils.data as data
import numpy as np
from torchvision.transforms import ToTensor, Scale, Compose, Pad, RandomHorizontalFlip, CenterCrop, RandomCrop, Scale 
from PIL import Image

import torch
import os

import pdb

VOX_CELEB_LOCATION = '/home/liyong/data/frame_cropped'

def load_img(file_path):
    img = Image.open(file_path).convert('RGB')
    return img

def load_img_mask(file_path):
    img = Image.open(file_path).convert('RGB')
    return img


class VoxCeleb2(data.Dataset):
    def __init__(self, num_views, random_seed, dataset, additional_face=True, jittering=False):
        if dataset == 1:
            self.ids = np.load('../Datasets/voxceleb1_ori/train.npy')
        if dataset == 2:
            self.ids = np.load('../Datasets/voxceleb1_ori/val.npy')
        if dataset == 3:
            self.ids = np.load('../Datasets/voxceleb1_ori/test.npy')
        self.rng = np.random.RandomState(random_seed)
        self.num_views = num_views
        #self.base_file = os.environ['VOX_CELEB_LOCATION'] + '/%s/' 
        self.base_file = VOX_CELEB_LOCATION + '/%s/' 
        crop = 200
        if jittering == True:
            precrop = crop + 24
            crop = self.rng.randint(crop, precrop)
            self.pose_transform = Compose([Scale((256,256)),
                Pad((20,80,20,30)),
                CenterCrop(precrop), RandomCrop(crop),
                Scale((256,256)), ToTensor()])
            self.transform = Compose([Scale((256,256)),
                Pad((24,24,24,24)),
                CenterCrop(precrop), 
                RandomCrop(crop),
                Scale((256,256)), ToTensor()])
        else:
            precrop = crop + 24
            self.pose_transform = Compose([Scale((256,256)),
                Pad((20,80,20,30)),
                CenterCrop(crop),
                Scale((256,256)), ToTensor()])
            self.transform = Compose([Scale((256,256)),
                Pad((24,24,24,24)),
                CenterCrop(precrop),
                Scale((256,256)), ToTensor()])

    def __len__(self):
        return self.ids.shape[0] - 1

    def __getitem__(self, index):
        #(other_face, _) = self.get_blw_item(self.rng.randint(self.__len__()))
        return self.get_blw_item(index)

    def get_blw_item(self, index):
        # Load the images
        imgs = [0] * (self.num_views)

        img_track = [d for d in os.listdir(self.base_file % self.ids[index]) if
                     os.path.isdir(self.base_file % self.ids[index] + '/' + d)]

        img_track_t = []
        while (len(img_track_t) == 0):
            img_video = img_track[self.rng.randint(len(img_track))]

            img_track_t = []
            img_track_t = [img_video + '/' + d for d in os.listdir(self.base_file % self.ids[index] + '/' + img_video)
                           if not (d == 'VISITED')]
        img_track = img_track_t[self.rng.randint(len(img_track_t))]

        img_faces = [d for d in os.listdir(self.base_file % self.ids[index] + '/' + img_track) if d[-4:] == '.jpg']

        if self.num_views > len(img_faces):
            img_index = self.rng.choice(range(len(img_faces)), self.num_views, replace=True)
        else:
            img_index = self.rng.choice(range(len(img_faces)), self.num_views, replace=False)

        img_name_list = []

        for i in range(0, self.num_views):
            img_name = self.base_file % self.ids[index] + '/' + img_track + '/' + img_faces[img_index[i]]
            img_name_list.append(img_name)
            imgs[i] = load_img(img_name)
            imgs[i] = self.transform(imgs[i])

        return imgs
        #return (img_name_list, imgs)

