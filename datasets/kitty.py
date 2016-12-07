import sys
import os
import glob
import numpy as np
from PIL import Image

import caffe


class kitty:
    def __init__(self, data_path):
        self.dir = data_path
        self.mean = np.array((72.78044, 83.21195, 73.45286), dtype=np.float32)
#        # import cityscapes label helper and set up label mappings
#        sys.path.insert(0, '{}/scripts/helpers/'.format(self.dir))
#        labels = __import__('labels')
#        labels_and_ids = [(l.name, l.trainId) for l in labels.labels if l.trainId >= 0 and l.trainId < 255]
#        self.classes = [l[0] for l in sorted(labels_and_ids, key=lambda x: x[1])]  # classes in ID order == network output order
#        self.id2trainId = {label.id: label.trainId for label in labels.labels}  # dictionary mapping from raw IDs to train IDs
#        self.trainId2color = {label.trainId: label.color for label in labels.labels}  # dictionary mapping train IDs to colors as 3-tuples
        self.classes = np.array([0, 1])

    def list_label_frames(self, split):
        def file2idx(f):
            """Helper to convert file path into frame ID"""
            city, shot, frame = (os.path.basename(f).split('_')[:3])
            return "_".join([city, shot, frame])
        frames = []
        scenes = [os.path.basename(f) for f in glob.glob('{}/data_road/training/{}/image_2/*'.format(self.dir, split))]
        for c in scenes:
#            files = sorted(glob.glob('{}/gtFine/{}/{}/*labelIds.png'.format(self.dir, split, c)))
            files = sorted(glob.glob('*.png'))
            frames.extend([file2idx(f) for f in files])
        return frames

    def load_image(self, split, scene, idx):
#        im = Image.open('{}/images/leftImg8bit/{}/{}/{}_leftImg8bit.png'.format(self.dir, split, city, idx))
        im = Image.open('{}/images/leftImg8bit/{}/{}/{}_leftImg8bit.png'.format(self.dir, split, scene, idx))
        return im

    def load_label(self):
        return np.array([ 0, 1])
    
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
        
#    def palette(self, label):
#        '''
#        Map trainIds to colors as specified in labels.py
#        '''
#        if label.ndim == 3:
#            label= label[0]
#        color = np.empty((label.shape[0], label.shape[1], 3))
#        for k, v in self.trainId2color.iteritems():
#            color[label == k, :] = v
#        return color