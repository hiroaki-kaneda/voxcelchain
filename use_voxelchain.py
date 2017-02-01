#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) 2017 BRILLIANTSERVICE CO.,LTD.
#
# This software is released under the MIT License.
# http://opensource.org/licenses/mit-license.php

import chainer
from chainer import serializers
import voxelchain
import numpy as np
import chainer.functions as F


def use_model(model):

    data = np.loadtxt('data/human_test_1_32_32_32.txt').reshape(1, 1, 32, 32, 32).astype(np.float32)
    y= model.fwd(data)
    A= F.softmax(y).data
    print(A.argmax(axis=1))
    print(A[0,A.argmax(axis=1)])

def main():
    model = voxelchain.VoxelChain()
    serializers.load_npz('result/VoxelChain.model',model)
    use_model(model)

if __name__ == '__main__':
    main()
