import numpy as np  
import random
import caffe

from lib import run_net

import matplotlib.pyplot as plt

import yaml

from datasets.kitty import kitty

random.seed(0xCAFFE)

def keep_layer(layer):
    """
    Filter layers to keep scores alone
    """
    is_split = 'split' in layer
    is_crop = 'c' == layer[-1]
    is_interp = 'up' in layer
    is_type = layer.startswith('score')
    return is_type and not (is_split or is_crop or is_interp)
    
def segsave(net, path):
    """
    Save maps to disk as compressed arrays.
    """
    for layer, blob in net.blobs.iteritems():
        if keep_layer(layer):
            np.savez('{}-{}.npz'.format(path, layer), blob.data[0])

#caffe.set_device(0)
#caffe.set_mode_cpu()


with open("config.yml", 'r') as ymlfile:
    path = yaml.load(ymlfile)['path']

CS = kitty('{}{}'.format(path, 'datasets/'))
n_cl = len(CS.classes)
split = 'training'

net = caffe.Net('{}{}'.format(path, 'nets/stage-voc-fcn8s.prototxt'),
                '{}{}'.format(path, 'nets/fcn8s-heavy-pascal.caffemodel'),
                caffe.TEST)

# pre processor for differencing
label_frames = CS.list_label_frames(split)
layers = filter(lambda l: keep_layer(l), net.blobs.keys())

# differences: layers, then argmaxes, and last is data and label
diffs = {}
zeros = np.zeros((len(label_frames)), dtype=np.float32)
diffs['data'] = zeros.copy()
diffs['label'] = zeros.copy()

for l in layers:
    diffs[l] = zeros.copy()
    diffs[l + '-argmax'] = zeros.copy()

for ix, frame_name in enumerate(label_frames):
    scene = frame_name.split('_', 1)[0]
    index = frame_name.split('_', 1)[1]
    im = CS.load_image(split, scene, index)    
    # handle first frame
    if (ix == 0):
        label = CS.load_label(split, scene, index)
        data = run_net.segrun(net, CS.preprocess(im))
        # TODO: hacking
        data[data == 0] = -1
        feats = [net.blobs[l].data[0].copy() for l in layers]
        argmaxes = [net.blobs[l].data[0].argmax(axis=0).copy() for l in layers if 'score' in l]
    else:
        new_label = CS.load_label(split, scene, index)
        new_data = run_net.segrun(net, CS.preprocess(im))
        new_feats = [net.blobs[l].data[0].copy() for l in layers]
        new_argmaxes = [net.blobs[l].data[0].argmax(axis=0).copy() for l in layers if 'score' in l]    
        # compute differences
        for lx, l in enumerate(layers):
            abs_diff = np.abs(new_feats[lx] - feats[lx])
            abs_avg = (np.abs(new_feats[lx]) + np.abs(feats[lx])) / 2.
            rel_diff = abs_diff / abs_avg
            rel_diff[np.isnan(rel_diff)] = 0
            diffs[l][ix] = rel_diff.mean()
            diffs[l + '-argmax'][ix] = np.array(new_argmaxes[lx] != argmaxes[lx]).mean()
        # TODO: hacking
        new_data[new_data == 0] = -1    
        diffs['data'][ix] = (np.abs(new_data - data) / ((np.abs(new_data) + np.abs(data)) / 2.)).mean()
        print np.unique(data)
        print np.unique(new_data)        
        
        diffs['label'][ix] = np.array(new_label != label).mean()
        # advance over old
        feats, argmaxes, data, label = new_feats, new_argmaxes, new_data, new_label    
        
    np.savez('{}/data_road/{}/image_2/{}/{}'.format(CS.dir, split, scene, ix), **diffs)

# calculate score #of different pixels / #of total pixels, output of the layer
# load frames

def load_layer_diffs(split, scene):
    diffs = np.load('{}/data_road/{}/image_2/{}/{}/diffs.npz'.format(CS.dir, split, scene))
    layers = diffs.keys()
    diffs = np.concatenate([d[..., np.newaxis] for l, d in diffs.iteritems()], axis=-1)
    return layers, diffs

# show mean differences across layers
#layers, diffs = load_layer_diffs('training', '02')
#for layer, diff in zip(layers, diffs.T):
#    print '{:<20}: {}'.format(layer, np.mean(diff))

#all_diffs = []
#for vid in sorted(CS.list_label_vids()):
#    for shot in CS.list_label_shots(vid):
#        layers, diffs = load_layer_diffs(vid, shot)
#        all_diffs.append(diffs)
#all_diff_arr = np.concatenate(all_diffs)

#means = np.zeros((len(all_diffs), len(layers)))
#for ix, diff in enumerate(all_diffs):
#    means[ix] = np.mean(diff, axis=0)

#for layer, mean in zip(layers, means.T):
#    print '{:<20}: {}'.format(layer, np.std(mean))

#vid = random.choice(CS.list_label_vids(class_))
#shot = random.choice(CS.list_label_shots(class_, vid))
#print class_, vid, shot
#
#layers, diffs = load_layer_diffs(vid, shot)
#diff_means = np.mean(diffs, axis=0)

#plot_layers = [l for l in layers if 'argmax' in l] + ['label']
#plot_ix = [layers.index(l) for l in plot_layers]

#plt.figure()
#plt.title('{} vid {} shot {}'.format(class_, vid, shot))
#plt.plot(diffs[:, plot_ix] - diff_means[plot_ix])
#plt.legend(plot_layers)
