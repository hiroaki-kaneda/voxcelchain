#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) 2017 BRILLIANTSERVICE CO.,LTD.
#
# This software is released under the MIT License.
# http://opensource.org/licenses/mit-license.php

from __future__ import print_function
import numpy as np
import chainer
from chainer import optimizers
from chainer import Chain
from chainer import training,iterators,serializers
from chainer.training import extensions
from chainer.datasets import tuple_dataset
import chainer.functions as F
import chainer.links as L
import subprocess

arg_batchsize = 10
arg_epoch = 20
#arg_iteration = 1000000
arg_test = True

# Define model
class VoxelChain(Chain):
    def __init__(self):
        super(VoxelChain, self).__init__(
            conv1 = L.ConvolutionND(3,  1, 20, 5), # 1 input, 20 outputs, filter size 5 pixels
            conv2 = L.ConvolutionND(3, 20, 20, 5), # 20 inputs, 20 outputs, filter size 5 pixels
            fc3=L.Linear(2500, 1300),
            fc4=L.Linear(1300, 10),
        )
        self.train = True

    def __call__(self, x, t):
        # To solve the classification problem with "softmax", use "softmax_cross_entropy".
        h = self.fwd(x)
        loss = F.softmax_cross_entropy (h, t)
        chainer.report({'loss': loss, 'accuracy': F.accuracy(h, t)}, self)
        return loss

    def fwd(self,x):
        h = F.max_pooling_nd(F.local_response_normalization(F.relu(self.conv1(x))), 3, stride=2)
        h = F.max_pooling_nd(F.local_response_normalization(F.relu(self.conv2(h))), 3, stride=2)
        h = F.dropout(F.relu(self.fc3(h)), train=self.train)
        h = self.fc4(h)
        return h


class TestModeEvaluator(extensions.Evaluator):

    def evaluate(self):
        model = self.get_target('main')
        model.train = False
        ret = super(TestModeEvaluator, self).evaluate()
        model.train = True
        return ret


def main():
    # Initialize model
    model = VoxelChain()             # Generate model
    chainer.cuda.get_device(0).use()  # Make a specified GPU current
    model.to_gpu()  # Copy the model to the GPU
    optimizer = optimizers.SGD()    # Selection of optimization algorithm
    optimizer.setup(model)          # Set model to algorithm

    # Load data
    data_list = np.array([['data/bicycle0000240_1_32_32_32.txt',240],   # 0
                 ['data/bicycle_man0000316_1_32_32_32.txt',316],        # 1
                 ['data/bike_post0000211_1_32_32_32.txt',211],          # 2
                 ['data/dog0000206_1_32_32_32.txt',206],                # 3
                 ['data/human0000207_1_32_32_32.txt',207],              # 4
                 ['data/mail_post0000232_1_32_32_32.txt',232],          # 5
                 ['data/motorcycle0000237_1_32_32_32.txt',237],         # 6
                 ['data/motorcycle_man0000247_1_32_32_32.txt',247],     # 7
                 ['data/pylon0000214_1_32_32_32.txt',214],              # 8
                 ['data/table0000210_1_32_32_32.txt',210]])             # 9

    list_w, list_l = data_list.shape
    print(data_list[0, 0],list_w,list_l)
    data = np.loadtxt(data_list[0, 0]).reshape(int(data_list[0, 1]), 1, 32, 32, 32).astype(np.float32)
    data = data[np.arange(200),:]
    print(data.shape)
    label = np.zeros(200).astype(np.int32)
    for i in range(1, list_w):
        print(data_list[i, 0])
        ld = np.loadtxt(data_list[i, 0]).reshape(int(data_list[i, 1]), 1, 32, 32, 32).astype(np.float32)
        ld = ld[np.arange(200),:]
        print(ld.shape)
        data = np.vstack([data, ld])
        ll = np.zeros(200).astype(np.int32) + i
        label = np.hstack([label, ll])


    # Divided into training data and test data
    N = label.size
    index = np.random.permutation(N)
    x_train = data[index[index % 5 != 0], :]
    y_train = label[index[index % 5 != 0]]
    x_test = data[index[index % 5 == 0], :]
    y_test = label[index[index % 5 == 0]]
    print("x_train: x_test:", x_train.shape, x_test.shape)
    print("y_train: y_test:", y_train.shape, y_test.shape)

    print(x_train)
    print(x_test)
    print(y_train)
    print(y_test)

    # Trainer must convert it to TupleDataset type
    train = tuple_dataset.TupleDataset(x_train, y_train.astype(np.int32))
    test = tuple_dataset.TupleDataset(x_test, y_test.astype(np.int32))

    # Iteration
    train_iter = iterators.SerialIterator(train, batch_size=arg_batchsize)
    test_iter = iterators.SerialIterator(test, batch_size=arg_batchsize, repeat=False,shuffle=False)

    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=0)
    trainer = training.Trainer(updater, (arg_epoch, 'epoch'), out='result')

    # Evaluate the model with the test dataset for each epoch

    val_interval = (10 if arg_test else 100000), 'iteration'
    log_interval = (10 if arg_test else 1000), 'iteration'
    trainer.extend(TestModeEvaluator(test_iter, model, device=0),
                   trigger=(arg_epoch, 'epoch'))

    # Dump
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(), trigger=val_interval)
    trainer.extend(extensions.snapshot_object(
        model, 'model_iter_{.updater.iteration}'), trigger=val_interval)
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.observe_lr(), trigger=log_interval)
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'main/loss', 'validation/main/loss',
        'main/accuracy', 'validation/main/accuracy', 'lr'
    ]), trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    # Run the training
    trainer.run()

    serializers.save_npz('result/VoxelChain.model', model)


if __name__ == '__main__':
    main()

    subprocess.call(["dot", "-Tpng", "result/cg.dot", "-o", "result/voxcelchain.png"])
