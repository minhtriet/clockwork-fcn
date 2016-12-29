import numpy as np
import random
import caffe

from lib import run_net

import yaml

from nets import kitty

from PIL import Image

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
caffe.set_mode_cpu()


with open("config.yml", 'r') as ymlfile:
    path = yaml.load(ymlfile)['path']

CS = kitty.kitty()
n_cl = len(CS.classes)
split = 'training'

net = caffe.Net('{}{}'.format(path, 'nets/stage-voc-fcn8s.prototxt'),
                '{}{}'.format(path, 'nets/fcn8s-heavy-pascal.caffemodel'),
                caffe.TEST)

# pre processor for differencing
vid = CS.list_vids()[0]
label_frames = CS.list_frames(vid)
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
    im = Image.open(frame_name)
    # handle first frame
    if (ix == 0):
        data = run_net.segrun(net, CS.preprocess(im))
        # TODO: hacking
        # data[data == 0] = -1
        feats = [net.blobs[l].data[0].copy() for l in layers]
        argmaxes = [net.blobs[l].data[0].argmax(axis=0).copy() for l in layers if 'score' in l]
    else:
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
        # new_data[new_data == 0] = -1
        diffs['data'][ix] = (np.abs(new_data - data) / ((np.abs(new_data) + np.abs(data)) / 2.)).mean()
        print np.unique(data)
        print np.unique(new_data)

        # advance over old
        feats, argmaxes = new_feats, new_argmaxes

    np.savez('{}/{}'.format(CS.dir, ix), **diffs)
