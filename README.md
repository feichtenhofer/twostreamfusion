================================================================================
# Convolutional Two-Stream Network Fusion for Video Action Recognition

This repository contains the code for our CVPR 2016 paper:

    Christoph Feichtenhofer, Axel Pinz, Andrew Zisserman
    "Convolutional Two-Stream Network Fusion for Video Action Recognition"
    in Proc. CVPR 2016

If you find the code useful for your research, please cite our paper:

        @inproceedings{feichtenhofer2016convolutional,
          title={Convolutional Two-Stream Network Fusion for Video Action Recognition},
          author={Feichtenhofer, Christoph and Pinz, Axel and Zisserman, Andrew},
          booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
          year={2016}
        }

# Requirements

The code was tested on Ubuntu 14.04 and Windows 10 using MATLAB R2015b and
 NVIDIA Titan X or Z GPUs. 

If you have questions regarding the implementation please contact:

    Christoph Feichtenhofer <feichtenhofer AT tugraz.at>

================================================================================

# Setup

1. Download the code ```git clone --recursive https://github.com/feichtenhofer/twostreamfusion```

2. Compile the code by running ```compile.m```.
    *  This will also compile a modified (and older) version of the 
[MatConvNet](http://www.vlfeat.org/matconvnet) toolbox. In case of any issues, 
please follow the [installation](http://www.vlfeat.org/matconvnet/install/) instructions on the
 MatConvNet [homepage](http://www.vlfeat.org/matconvnet).

3. Edit the file cnn_setup_environment.m to adjust the models and data paths.

4. Download pretrained model files and the datasets, linked below and unpack them into your models/data directory.
* Optionally you can pretrain your own twostream models by running
    1. `cnn_ucf101_spatial();` to train the appearance network stream.
    1. `cnn_ucf101_temporal();` to train the optical flow network stream.

5. Run
`cnn_ucf101_fusion();` this will use the downloaded models and demonstrate training of our final architecture on UCF101/HMDB51. 
    - In case you would like to train on the CPU, clear the variable `opts.train.gpus`
    - In case you encounter memory issues on your GPU, consider decreasing the `cudnnWorkspaceLimit` (512MB is default)

# Pretrained models
- Download our baseline networks trained on UCF101 here:
    - [VGG-16](http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/models/twostream_base/vgg16/)
    - [ResNet-50](http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/models/twostream_base/resnet50/)
    - [ResNet-152](http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/models/twostream_base/resnet152/)

# Data
Pre-computed optical flow images and resized rgb frames for the [UCF101](http://crcv.ucf.edu/data/UCF101.php) and [HMDB51](http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/) datasets
- UCF101 RGB: [part1](http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_jpegs_256.zip.001)
[part2](http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_jpegs_256.zip.002)
[part3](http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_jpegs_256.zip.003)

- UCF101 Flow: [part1](http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_tvl1_flow.zip.001)
[part2](http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_tvl1_flow.zip.002)
[part3](http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_tvl1_flow.zip.003)

- HMDB51 RGB: [part1](http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/hmdb51_jpegs256.zip)
- HMDB51 Flow: [part1](http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/hmdb51_tvl1_flow.zip)

# Use it on your own dataset
- Our [Optical flow extraction tool](https://github.com/feichtenhofer/gpu_flow) provides OpenCV wrappers for optical flow extraction on a GPU.
