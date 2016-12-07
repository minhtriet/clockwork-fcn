import numpy as np
import random
import caffe

from lib import run_net

import matplotlib
import matplotlib.pyplot as plt

from lib import score_util

from datasets.kitty import kitty

random.seed(0xCAFFE)

caffe.set_device(0)
caffe.set_mode_cpu()

CS = kitty('/media/remote_home/mtriet/clockwork-fcn/datasets/')
#CS = kitty('/home/minhtriet/CS/lab/clockwork-fcn/datasets/')
n_cl = len(CS.classes)
split = 'training'
scene= '02'
abel_frames = CS.list_label_frames(split)

net = caffe.Net('/media/remote_home/mtriet/clockwork-fcn/nets/stage-voc-fcn8s.prototxt',
                '/media/remote_home/mtriet/clockwork-fcn/nets/fcn8s-heavy-pascal.caffemodel',
                caffe.TEST)

hist_perframe = np.zeros((n_cl, n_cl))
label_frames = CS.list_label_frames(split)
for i, idx in enumerate(label_frames):
    if i % 100 == 0:
        print 'running {}/{}'.format(i, len(label_frames))
    scene = idx.split('_', 1)[0]
    index = idx.split('_', 1)[1]
    # idx is scene_shot_frame
    im = CS.load_image(split, scene, index)
    out = run_net.segrun(net, CS.preprocess(im))
    label = CS.load_label()
    hist_perframe += score_util.fast_hist(label.flatten(), out.flatten(), n_cl)
    print hist_perframe
    
idx = 0
im = CS.load_image(split, scene, idx)
out = run_net.segrun(net, CS.preprocess(im))
segshow(im, CS.palette(label), CS.palette(out))
plt.figure()
plt.imshow(out)
plt.axis('off')
plt.tight_layout()
plt.show()

# calculate score #of different pixels / #of total pixels, output of the layer
# load frames

