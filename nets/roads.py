import os
import glob
import numpy as np

import PIL
from PIL import Image

import pdb

class Road:
    def __init__(self, data_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))):
        self.dir = os.path.join(data_path, 'datasets')
        self.mean = (104.00698793, 116.66876762, 122.67891434) # imagenet mean
        self.MAX_DIM = 500.0  # match PASCAL VOC training data
        self.label_thresh = 15
        self.classes = ['road']

    def list_vids(self):
        scenes = [os.path.basename(f) for f in glob.glob('{}/road/*'.format(self.dir))]
        return scenes 

    def list_frames(self, vid):
        f = [os.path.basename(f) for f in glob.glob('{}/road/{}/image_03/data/*'.format(self.dir, vid))]
        return f

    def load_image(self, vid, idx):
        im = Image.open('{}/road/{}/image_03/data/{}'.format(self.dir, vid, idx))
        im = self.resize(im, False)
        return im

    def resize(self, im, label=False):
        dims = np.array(im).shape
        if len(dims) > 2:
            dims = dims[:-1]
        max_val, max_idx = np.max(dims), np.argmax(dims)
        scale = self.MAX_DIM / max_val
        new_height, new_width = int(dims[0]*scale), int(dims[1]*scale)
        if label:
            im = im.resize((new_width, new_height), resample=PIL.Image.NEAREST)
        else:
            im = im.resize((new_width, new_height), resample=PIL.Image.BILINEAR)
        return im

    def preprocess(self, im):
        """
        Preprocess loaded image (by load_image) for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        in_ = Image.new("RGB", im.size)
        in_.paste(im)
        in_ = np.array(in_, dtype=np.float32)
        in_ = in_[:, :, ::-1]
        in_ -= self.mean
        in_ = in_.transpose((2, 0, 1))
        return in_

    def palette(self, label_im):
        '''
        Transfer the VOC color palette to an output mask
        '''
        if label_im.ndim == 3:
            label_im = label[0]
        label = Image.fromarray(label_im, mode='P')
        label.palette = copy.copy(self.voc_palette)
        return label

    def to_voc_label(self, label, class_, voc_classes):
        label = np.array(label, dtype=np.uint8)
        label[label <= self.label_thresh] = 0
        label[label > self.label_thresh] = voc_classes.index(class_)

        return label

