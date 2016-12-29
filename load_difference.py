#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 20:06:58 2016
Load npz and show difference
@author: minhtriet
"""

import numpy as np


import yaml

import ipdb

from nets.kitty import kitty

import matplotlib.pyplot as plt

with open("config.yml", 'r') as ymlfile:
    path = yaml.load(ymlfile)['path']

CS = kitty('{}{}'.format(path, 'datasets/'))

def load_layer_diffs(vid):
    diffs = np.load('{}/data_road/training/{}.npz'.format(CS.dir, vid))
    diffs.files.remove('score_pool3-argmax')
    diffs.files.remove('data')
    diffs.files.remove('label')
    diffs.files.remove('score-argmax')
    diffs.files.remove('score_fr-argmax')
    diffs.files.remove('score_pool4-argmax')
    diffs.files.remove('score')
    diffs.files.remove('score_pool4')
    layers = diffs.keys()
    diffs = np.concatenate([d[..., np.newaxis] for l, d in diffs.iteritems()], axis=-1)
    return layers, diffs

all_diffs = []
for vid in [1]:
    layers, diffs = load_layer_diffs(vid)
    diff_means = np.mean(diffs, axis=0)
    all_diffs.append(diffs)
    #plot_layers = [l for l in layers if 'argmax' in l]
    plot_layers = [l for l in layers]
    plot_ix = [layers.index(l) for l in plot_layers]
    #ipdb.set_trace()
    plt.plot(diffs[:, plot_ix] - diff_means[plot_ix])
    plt.legend(plot_layers)

    plt.savefig('{}/data_road/training/graph{}.png'.format(CS.dir, vid))
all_diff_arr = np.concatenate(all_diffs)

means = np.zeros((len(all_diffs), len(layers)))
for ix, diff in enumerate(all_diffs):
    means[ix] = np.mean(diff, axis=0)

