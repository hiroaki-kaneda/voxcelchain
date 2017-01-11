# voxcelchain
3-D Convolution Neural Network using the Chainer

## What is VoxcelChain
VoxcelChain is a three-dimensional shape general object recognition program using deep learning. VoxcelChain is a 3D Convolution Neural Network, which is a feedforward neural network in which two types of layers, a convolution layer and a pooling layer, are alternately stacked.

![conceptual structure ](https://github.com/hiroaki-kaneda/voxcelchain/blob/master/doc/voxcel_chain.png)

## Installation
`voxcelchain` is based on [python](https://www.python.org/)
and [Chainer](http://chainer.org/). Then we also need [CUDA](https://developer.nvidia.com/cuda-downloads).

### Setup CUDA8
Setup [CUDA](https://developer.nvidia.com/cuda-downloads).

### Installing Python3 & chainer on Ubuntu 16.04
Ubuntu 16.04 ships with Python 3 pre-installed.
```sh
sudo apt-get update
sudo apt-get -y upgrade
```

Install chainer and numpy.
```sh
sudo apt-get install python3-pip
sudo pip3 install chainer
sudo pip3 install numpy
```

### Get voxcelchain
You can do something like
```sh
git clone https://github.com/hiroaki-kaneda/voxcelchain.git
```

## Usage
![people](https://github.com/hiroaki-kaneda/voxcelchain/blob/master/doc/voxcel_result.png)

It contains sample datasets of 10 classes, so we just run the program. The sample datasets are in the `data` directory. The data is divided into classes, and it is a one-dimensional array. Read the data and reshape it so Chainer can handle it. Learning begins when you do this.

Execute as below:
```sh
python3 voxelchain.py
```

You can see the progress at the console.

When learning is completed, model data is saved as `VoxcelChain.model` in the `result` directory.

### Details

In learning point cloud data, a point cloud of only that object is required. Since the point cloud obtained from the sensor includes walls and floors as well as objects other than the object, it is necessary to cut out only the object.

We made a tool, it cut out an object from point cloud. The tool is divided into two phases.

* __Point Cloud capture tool__ This is a process of capturing point clouds and storing them in a file. For example, it is equivalent to taking a picture.
* __Object segmentation classification tool__ This opens the captured point cloud file, and you click on an object to cut out only that object.

The data cut out by this tool is converted into 0,1 data shrinked in units of voxcel. Processing is as follows
* Prepare 32x32x32 voxcel and array[32][32][32] to be array[x][y][z].
* 1 if　there is data, 0 if there is data.
* The output array is one-dimensional array data.

The sample dataset contains multiple objects in one file. For example, `human0000207_1_32_32_32.txt` contains data of 207 objects. Each has a one channel and it is 32x32x32 voxcel size.

Like image recognition in deep learning, Convolution Neural Network(CNN) is expected to be effective as well. Therefore, this time, We have constructed CNN which learn 3D point cloud obtained from sensor, and learned by giving data of 10 categories.
200 data were prepared for each category. To obtain point cloud data, we rendered objects in 3D software, simulated Kinect sensor data, and extracted point cloud data from it, not from actual sensors.

3D simulation uses Gazebo. We simulate the sensor value obtained at the actual position to attach the sensor to the robot.

## Link
For more information, visit our [website](http://bril-tech.blogspot.jp/2017/01/VoxcelChain-3D-Convolutional-Neural-Networks.html).
