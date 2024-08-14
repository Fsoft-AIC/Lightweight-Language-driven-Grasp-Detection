import glob
import os
import re

import pickle
import torch

from utils.dataset_processing import grasp, image, mask
from .grasp_data import GraspDatasetBase
import numpy as np
import clip
from torchvision import transforms
class GraspAnythingDataset(GraspDatasetBase):
    """
    Dataset wrapper for the Grasp-Anything dataset.
    """

    def __init__(self, file_path, ds_rotate=0, **kwargs):
        """
        :param file_path: Grasp-Anything Dataset directory.
        :param ds_rotate: If splitting the dataset, rotate the list of items by this fraction first
        :param kwargs: kwargs for GraspDatasetBase
        """
        super(GraspAnythingDataset, self).__init__(**kwargs)

        self.grasp_files = glob.glob(os.path.join(file_path, 'grasp_label', '*.pt'))
        self.prompt_files = glob.glob(os.path.join(file_path, 'grasp_instructions', '*.pkl'))
        self.rgb_files = glob.glob(os.path.join(file_path, 'image', '*.jpg'))
        self.part_masks = glob.glob(os.path.join(file_path, 'part_mask', '*.npy'))
        # self.mask_files = glob.glob(os.path.join(file_path, 'mask', '*.npy'))

        self.grasp_files.sort()
        self.prompt_files.sort()
        self.rgb_files.sort()
        self.part_masks.sort()
        # self.mask_files.sort()

        self.length = len(self.grasp_files)

        if self.length == 0:
            raise FileNotFoundError('No dataset files found. Check path: {}'.format(file_path))

        if ds_rotate:
            self.grasp_files = self.grasp_files[int(self.length * ds_rotate):] + self.grasp_files[
                                                                                 :int(self.length * ds_rotate)]
            

    def _get_crop_attrs(self, idx):
        gtbbs = grasp.GraspRectangles.load_from_grasp_anything_file(self.grasp_files[idx])
        center = gtbbs.center
        left = max(0, min(center[1] - self.output_size // 2, 416 - self.output_size))
        top = max(0, min(center[0] - self.output_size // 2, 416 - self.output_size))
        return center, left, top

    def get_gtbb(self, idx, rot=0, zoom=1.0):       
        gtbbs = grasp.GraspRectangles.load_from_grasp_anything_file(self.grasp_files[idx], scale=self.output_size / 416.0)

        c = self.output_size // 2
        gtbbs.rotate(rot, (c, c))
        gtbbs.zoom(zoom, (c, c))
        return gtbbs

    def get_depth(self, idx, rot=0, zoom=1.0):
        depth_img = image.DepthImage.from_tiff(self.depth_files[idx])
        center, left, top = self._get_crop_attrs(idx)
        depth_img.rotate(rot, center)
        depth_img.crop((top, left), (min(480, top + self.output_size), min(640, left + self.output_size)))
        depth_img.normalise()
        depth_img.zoom(zoom)
        depth_img.resize((self.output_size, self.output_size))
        return depth_img.img

    def get_rgb(self, idx, rot=0, zoom=1.0, normalise=True):
        rgb_file = re.sub(r'_[0-9]+_[0-9]+\.pt$', '.jpg', self.grasp_files[idx])
        rgb_file = rgb_file.replace("grasp_label", "image")
        rgb_img = image.Image.from_file(rgb_file)

        rgb_img.rotate(rot)
        rgb_img.zoom(zoom)
        rgb_img.resize((self.output_size, self.output_size))
        if normalise:
            rgb_img.normalise()
            rgb_img.img = rgb_img.img.transpose((2, 0, 1))
        return rgb_img.img
    
    def get_text_embed(self, idx):
        with open(self.prompt_files[idx], 'rb') as f:
            text = pickle.load(f)
        
        text = text.split(' ')
        text_embed = clip.tokenize(text)
        return text_embed
    
    def get_mask(self, idx):
        mask_file = self.grasp_files[idx].replace(".pt", ".npy")
        mask_file = mask_file.replace("grasp_label", "part_mask")
        mask_img = np.load(mask_file)
        mask_img = mask_img*255
        mask_transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Resize((224, 224)), # change to (336,336) when using ViT-L/14@336px
            transforms.Normalize(0.5, 0.26)
        ])
        
        alpha = mask_transform(mask_img)
        
        return alpha