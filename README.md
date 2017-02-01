
# voxcelchain
3-D Convolutional Neural Network using the Chainer

![conceptual structure ](https://github.com/hiroaki-kaneda/voxcelchain/blob/master/doc/voxcel_chain.png)


## What is VoxcelChain
VoxcelChain is a three-dimensional shape general object recognition program using deep learning. VoxcelChain is a 3D Convolution Neural Network, which is a feedforward neural network in which two types of layers, a convolution layer and a pooling layer, are alternately stacked.

## Installation
`voxcelchain` is based on [python](https://www.python.org/)
and [Chainer](http://chainer.org/). And [CUDA](https://developer.nvidia.com/cuda-downloads) is required as well.

## Requirements
This module requires the following modules:

* Python 3.5.2
* numpy 1.11.2
* Chainer 1.18.0
* CUDA V8.0.44

This has been tested on Ubuntu 16.04.

### Setup CUDA8
Setup [CUDA](https://developer.nvidia.com/cuda-downloads).

### Installing Python3 & Chainer on Ubuntu 16.04
Python 3 should be pre-installed in Ubuntu 16.04.
```sh
sudo apt-get update
sudo apt-get -y upgrade
```

Install Chainer and numpy.
```sh
sudo apt-get install python3-pip
sudo pip3 install chainer
sudo pip3 install numpy
```

### Get voxcelchain
An example what you may do is:
```sh
git clone https://github.com/hiroaki-kaneda/voxcelchain.git
```

## Usage
![people](https://github.com/hiroaki-kaneda/voxcelchain/blob/master/doc/voxcel_result.png)

 Since it contains sample datasets of 10 classes, you just run the program. The sample datasets are in the `data` directory. The data will be divided into each class, and it should be a one dimensional array.  It will read the data and reshape it so that Chainer will be able to handle it.  Once this is done, learning will start.

Execute as below:
```sh
python3 voxelchain.py
```

You can see the progress at the console.

When learning is completed, model data will be saved as `VoxcelChain.model` in the `result` directory.

### Details

In learning point cloud data, a point cloud of only that object is required. Since the point cloud obtained from the sensor includes walls and floors as well as objects other than the object, it is necessary to cut out only the object.

We made a tool to cut out an object from point cloud. There are two phases for the tool.

* __Point Cloud capture tool__ This is a process of capturing point clouds and storing them in a file. For example, it is equivalent to taking a picture.
* __Object segmentation classification tool__ This opens the captured point cloud file, and click on an object to cut out only that object.

The data which cut out by this tool will be converted into 0,1 data shrinked in units of voxcel. Processing is as follows
* Prepare 32x32x32 voxcel and array[32][32][32] to be array[x][y][z].
* 1 if there is data, 0 if there is data.
* The output array should be one-dimensional array data.

The sample dataset contains multiple objects in one file. e.g. `human0000207_1_32_32_32.txt` contains data of 207 objects. Each has a one channel and it is 32x32x32 voxcel size.

As well as an image recognition in deep learning, Convolutional Neural Network(CNN) is also expected to be effective. Therefore, this time, We have constructed CNN which learn 3D point cloud obtained from sensor, and learned by giving data of 10 categories.
There are 200 data for each category. To obtain point cloud data, we rendered objects in 3D software, simulated Kinect sensor data, and extracted point cloud data from it, not from actual sensors.

Gazebo is used for 3D simulation. We have simulated the sensor value which has been obtained from actual position in order to attach the sensor to the robot.

## Link
For more information, visit our [website](http://bril-tech.blogspot.com/2017/02/VoxcelChain-3D-Convolutional-Neural-Networks.html).
