import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
import random
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from util import custom_transforms
import torchvision.transforms as transforms
import pdb
import torch
import cv2
from util.smart_crop_transforms import RandomCropDramaticlly
from util import augment

class RetrievalWorkshopBaselineEval2DDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        
        # 2d features
        self.shapes_info = os.listdir(os.path.join(opt.dataroot, 'input_data'))
        if '.DS_Store' in self.shapes_info: self.shapes_info.remove('.DS_Store')
        self.shapes_info.sort()

        self.phase = opt.phase
        self.data_size = len(self.shapes_info)
        
        self.input_2d_path = os.path.join(opt.dataroot, 'input_data')
        #self.input_3d_path = '/home/dh/zdd/data/test/render5_black/' #os.path.join(opt.dataroot, 'notexture_pool_data')
        self.input_3d_path = os.path.join(opt.dataroot, 'render5_black')
        

        self.query_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        opt.fine_size = 256
        self.fine_size = opt.fine_size
        self.phase = opt.phase
        #self.query_random_erase = transforms.Compose([transforms.RandomErasing()])

    def __getitem__(self, index):
        shape_info = self.shapes_info[index % self.data_size]
    
        # 2d feature data
        image_name = shape_info
        query_img = self._load_2d_image(image_name)
        
        center_label = 0
        cate_label = 0
        view_label = 0
        return {'image_name': image_name.split('.')[0], 'query_img':query_img, 'center_label':center_label, 'cate_label':cate_label}


    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return self.data_size
 
    def _load_3d_image(self, shape_id, image_name):
        #img_file = os.path.join(self.input_3d_path, shape_id, image_name.split('.')[0] + '.jpg')
        img_file = os.path.join(self.input_3d_path, shape_id, image_name)
        img = Image.open(img_file).convert('RGB') 
        trans_img = self.query_transform(img)
        return trans_img
    
    def _load_2d_image(self, image_name):
        img_file = os.path.join(self.input_2d_path, image_name)
        img = Image.open(img_file).convert('RGB')
        trans_img = self.query_transform(img)
        return trans_img

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor
