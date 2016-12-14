#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 20:06:58 2016
Load npz and show difference
@author: minhtriet
"""

import numpy as np  

import matplotlib.pyplot as plt

import yaml

from datasets.kitty import kitty

with open("config.yml", 'r') as ymlfile:
    path = yaml.load(ymlfile)['path']

CS = kitty('{}{}'.format(path, 'datasets/'))

def load_layer_diffs(vid):
    diffs = np.load('{}/data_road/training/image_2/{}/diff.npz'.format(CS.dir, vid))
    layers = diffs.keys()
    diffs = np.concatenate([d[..., np.newaxis] for l, d in diffs.iteritems()], axis=-1)
    return layers, diffs

# show mean differences across layers
#layers, diffs = load_layer_diffs('training', '02')
#for layer, diff in zip(layers, diffs.T):
#    print '{:<20}: {}'.format(layer, np.mean(diff))

all_diffs = []
for vid in sorted(CS.list_vids()):
    layers, diffs = load_layer_diffs(vid)
    diff_means = np.mean(diffs, axis=0)
    all_diffs.append(diffs)
    plot_layers = [l for l in layers if 'argmax' in l] + ['label']
    plot_ix = [layers.index(l) for l in plot_layers]
    plt.figure()
    plt.title('vid {}'.format(vid))
    plt.plot(diffs[:, plot_ix] - diff_means[plot_ix])
    plt.legend(plot_layers)
    plt.savefig('{}/data_road/training/image_2/{}/graph.pdf'.format(CS.dir, vid))
all_diff_arr = np.concatenate(all_diffs)

means = np.zeros((len(all_diffs), len(layers)))
for ix, diff in enumerate(all_diffs):
    means[ix] = np.mean(diff, axis=0)

#vid = random.choice(CS.list_label_vids(class_))
#shot = random.choice(CS.list_label_shots(class_, vid))
#print class_, vid, shot
