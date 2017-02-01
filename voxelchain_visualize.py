#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) 2017 BRILLIANTSERVICE CO.,LTD.
#
# This software is released under the MIT License.
# http://opensource.org/licenses/mit-license.php

from chainer import serializers
import voxelchain
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.cm as cm
import chainer.functions as F
import json



def conv2(model):
    n1, n2, x, y, z = model.conv2.W.shape
    for nn in range(0, n1):
        fig = plt.figure()
        for mm in range(0, n2):
            ax = fig.add_subplot(4, 5, mm + 1, projection='3d')
            ax.set_xlim(0.0, x)
            ax.set_ylim(0.0, y)
            ax.set_zlim(0.0, z)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])

            for xx in range(0, x):
                for yy in range(0, y):
                    for zz in range(0, z):
                        max = np.max(model.conv2.W.data[nn, mm:])
                        min = np.min(model.conv2.W.data[nn, mm:])
                        step = (max - min) / 1.0
                        C = (model.conv2.W.data[nn, mm, xx, yy, zz] - min) / step
                        color = cm.cool(C)
                        C = abs(1.0-C)
                        ax.plot(np.array([xx]), np.array([yy]), np.array([zz]), "o", color=color, ms=7.0 * C, mew=0.1)

        plt.savefig("result/graph_conv2_" + str(nn) +  ".png")


def conv1(model):
    n1, n2, x, y, z = model.conv1.W.shape
    fig = plt.figure()
    for nn in range(0, n1):
        ax = fig.add_subplot(4, 5, nn+1, projection='3d')
        ax.set_xlim(0.0, x)
        ax.set_ylim(0.0, y)
        ax.set_zlim(0.0, z)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        for xx in range(0, x):
            for yy in range(0, y):
                for zz in range(0, z):
                    max = np.max(model.conv1.W.data[nn, :])
                    min = np.min(model.conv1.W.data[nn, :])
                    step = (max - min) / 1.0
                    C = (model.conv1.W.data[nn, 0, xx, yy, zz] - min) / step
                    color = cm.cool(C)
                    C = abs(1.0 - C)
                    ax.plot(np.array([xx]), np.array([yy]), np.array([zz]), "o", color=color, ms=7.0*C, mew=0.1)

    plt.savefig("result/graph_conv1.png")

def create_graph():
    logfile = 'result/log'
    xs = []
    ys = []
    ls = []
    f = open(logfile, 'r')
    data = json.load(f)

    print(data)

    for d in data:
        xs.append(d["iteration"])
        ys.append(d["main/accuracy"])
        ls.append(d["main/loss"])

    plt.clf()
    plt.cla()
    plt.hlines(1, 0, np.max(xs), colors='r', linestyles="dashed")  # y=-1, 1に破線を描画
    plt.title(r"loss/accuracy")
    plt.plot(xs, ys, label="accuracy")
    plt.plot(xs, ls, label="loss")
    plt.legend()
    plt.savefig("result/log.png")

def main():
    model = voxelchain.VoxelChain()
    serializers.load_npz('result/VoxelChain.model',model)
    conv1(model)
    conv2(model)
    create_graph()


if __name__ == '__main__':
    main()
