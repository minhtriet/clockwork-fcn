import os
import glob
import numpy as np
from PIL import Image

class kitty:
    def __init__(self, data_path):
        self.dir = data_path
        self.mean = np.array((72.78044, 83.21195, 73.45286), dtype=np.float32)
        self.classes = np.array([-1, 1])

    def list_label_frames(self, split):
        def file2idx(scene, f):
            """Helper to convert file path into frame ID"""
            typ, frame = (os.path.basename(f).split('_')[:2])
            return "_".join([scene, typ, frame])
        frames = []
        scenes = [os.path.basename(f) for f in glob.glob('{}/data_road/{}/image_2/*'.format(self.dir, split))]
        for c in scenes:
#            files = sorted(glob.glob('{}/gtFine/{}/{}/*labelIds.png'.format(self.dir, split, c)))
            files = sorted(glob.glob('{}/data_road/{}/image_2/{}/*.png'.format(self.dir, split, c)))
            frames.extend([file2idx(c, f) for f in files])
        return frames

    def load_image(self, split, scene, idx):
        im = Image.open('{}/data_road/{}/image_2/{}/{}'.format(self.dir, split, scene, idx))
        return im

    def load_label(self, split, scene, idx):
        idx = idx.split('_')
        idx.insert(1,'lane')
        im = Image.open('{}/data_road/{}/gt_image_2/{}/{}'.format(self.dir, split, scene, '_'.join(idx)))
        im = np.array(im, dtype=np.uint8)
        im = im[np.newaxis, ...]
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
