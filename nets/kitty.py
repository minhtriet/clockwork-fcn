import os
import glob
import numpy as np
from PIL import Image

class kitty:
    def __init__(self, data_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))):
        self.dir = data_path
        self.mean = (104.00698793, 116.66876762, 122.67891434) # imagenet mean
        self.train_vids = self.list_label_frames("training")

    def list_vids(self):
        path = '{}/datasets/kitti/'.format(self.dir)
        return [ os.path.join(path, name) for name in os.listdir(path) ]

    def list_frames(self, vid):
        frames = []
        path = '{}/{}_drive_0002_sync/image_03/data/'.format(vid,
                os.path.basename(vid))
        files = sorted(glob.glob('{}/*.png'.format(path)))
        frames.extend(files)
        return frames

    def list_label_frames(self, split):
        def file2idx(scene, f):
            """Helper to convert file path into frame ID"""
            typ, frame = (os.path.basename(f).split('_')[:2])
            return "_".join([scene, typ, frame])
        frames = []
        scenes = [os.path.basename(f) for f in glob.glob('{}/data_road/{}/image_2/*'.format(self.dir, split))]
        for c in scenes:
            files = sorted(glob.glob('{}/data_road/{}/image_2/{}/*.png'.format(self.dir, split, c)))
            frames.extend([file2idx(c, f) for f in files])
        return frames

    def load_image(self, split, scene, idx):
        im = Image.open('{}/data_road/{}/image_2/{}/{}'.format(self.dir, split, scene, idx))
        return im

    def change_color(im, origin_color, new_color):
        im = im.convert('RGBA')

        data = np.array(im)   # "data" is a height x width x 4 numpy array
        red, green, blue, alpha = data.T # Temporarily unpack the bands for readability

        # Replace white with red... (leaves alpha values alone...)
        areas = (red == origin_color[0]) & (blue == origin_color[1]) & (green == origin_color[2])
        data[..., :-1][white_areas.T] = new_color # Transpose back needed

        im2 = Image.fromarray(data)
        return im2 


    def load_label(self, split, scene, idx):
        idx = idx.split('_')
        idx.insert(1,'lane')
        im = Image.open('{}/data_road/{}/gt_image_2/{}/{}'.format(self.dir, split, scene, '_'.join(idx)))
        # change color to fit
        im = change_color(im, (255, 0, 0), (0, 0, 0))
        im = change_color(im, (255, 0, 255), (0,64,128))
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
        label = label[np.newaxis, ...]
        label[label <= self.label_thresh] = 0
        label[label > self.label_thresh] = voc_classes.index(class_)
        # everything is background now, street labeled 21
        label[label < 21] = 0        
        return label
