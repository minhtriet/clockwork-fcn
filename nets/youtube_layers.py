#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 23:16:49 2016

@author: minhtriet
"""

import caffe

import numpy as np
from PIL import Image

import os
import ipdb
from youtube import youtube

YT = youtube("{}/datasets".format(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import random

class YoutubeSegDataLayer(caffe.Layer):

    """
    Load (input image, label image) pairs from Kitti
    one-at-a-time while reshaping the net to preserve dimensions.
    Use this to feed data to a fully convolutional network.
    """

    def setup(self, bottom, top):
        """
        Setup data layer according to parameters:
        - dir: path to PASCAL VOC year dir
        - split: train / val / test
        - mean: tuple of mean values to subtract
        - randomize: load in random order (default: True)
        - seed: seed for randomization (default: None / current time)
        for PASCAL VOC semantic segmentation.
        example
        params = dict(dir="/path/to/PASCAL/VOC2011",
            mean=(104.00698793, 116.66876762, 122.67891434),
            split="val")
        """
        # config
        params = eval(self.param_str)
        self.dir = YT.dir
        self.split = params['split']
        self.mean = np.array(params['mean'])
        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)

        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        self.indices = YT.load_dataset()
        self.idx = 0

        # make eval deterministic
        if 'train' not in self.split:
            self.random = False

        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            self.idx = self.indices[random.randint(0, len(self.indices) - 1)]
            frames = YT.list_label_frames(self.idx[0], self.idx[1], self.idx[2])
            self.idx = self.idx + (frames[random.randint(0, len(frames) - 1)], )

    def reshape(self, bottom, top):
        # load image
        im = YT.load_frame(self.idx[0], self.idx[1], self.idx[2],
                self.idx[3])
        in_ = np.array(im, dtype=np.float32)
        in_ = in_[:,:,::-1]
        in_ -= self.mean
        self.data = in_.transpose((2,0,1))
        # load label
        im = YT.load_label(self.idx[0], self.idx[1], self.idx[2],
                self.idx[3])
        im = im.convert('P', palette=Image.ADAPTIVE, colors=3)
        self.label = np.array(im, dtype=np.uint8)
        self.label = self.label[np.newaxis, ...]
        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(1, *self.data.shape)
        top[1].reshape(1, *self.label.shape)


    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data
        top[1].data[...] = self.label

        # pick next input
        if self.random:
            self.idx = random.randint(0, len(self.indices)-1)
        else:
            self.idx += 1
            if self.idx == len(self.indices):
                self.idx = 0


    def backward(self, top, propagate_down, bottom):
        pass


    def load_image(self, idx):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        im = Image.open("{}/youtube_masks/{}/data/{}shots/{}/images/{}".format(YT.dir, idx[0],
            idx[1], idx[2], idx[3]))
        return in_


    def load_label(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        im = Image.open("{}/youtube_masks/{}/data/{}shots/{}/labels/{}".format(YT.dir, idx[0],
            idx[1], idx[2], idx[3]))
        return label
