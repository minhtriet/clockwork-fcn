#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 21:34:35 2016

@author: minhtriet
"""

import caffe
import surgery #, score
    
#import numpy as np
import os

path = os.path.dirname(os.path.realpath(__file__))
weights = '{}/nets/fcn8s-heavy-pascal.caffemodel'.format(path)

# init
#caffe.set_device(int(sys.argv[1]))
#caffe.set_mode_gpu()
caffe.set_mode_cpu()

solver = caffe.SGDSolver('{}/nets/kitti_solver.prototxt'.format(path))
solver.net.copy_from(weights)

# surgeries
#interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
#surgery.interp(solver.net, interp_layers)

# scoring
#val = np.loadtxt('../data/segvalid11.txt', dtype=str)
#
#for _ in range(25):
#    solver.step(4000)
#    score.seg_tests(solver, False, val, layer='score')