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

2. Compile the code by ```running compile.m```.
- This will also compile a modified (and older) version of the 
[MatConvNet](http://www.vlfeat.org/matconvnet) toolbox. In case of any issues, 
please follow the [installation](http://www.vlfeat.org/matconvnet/install/) instructions on the
 MatConvNet [homepage](http://www.vlfeat.org/matconvnet).

3. Download pretrained model files and the datasets, linked below and unpack them into your models/data directory.

4. Edit the file cnn_setup_environment.m to adjust the models and data paths.

5. Run
``` cnn_ucf101_fusion(); ``` this will use the downloaded models and demonstrate training of our final architecture on UCF101. 
- In case you would like to train on the CPU, clear the variable ```opts.train.gpus```
- In case you encounter memory issues on your GPU, consider decreasing the ```cudnnWorkspaceLimit``` (512MB is default)

# Pretrained models
- Spatial network for UCF101 [(split1)](http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/models_twostream/ucf101-img-vgg16-split1-dr0.85.mat)
- Temporal network for UCF101 [(split1)](http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/models_twostream/ucf101-TVL1flow-vgg16-split1-dr0.9.mat)

# Data
Pre-computed optical flow images and resized rgb frames for the [UCF101](http://crcv.ucf.edu/data/UCF101.php) and [HMDB51](http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/) datasets
- UCF101 RGB: [part1](http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_jpegs_256.zip.001)
[part2](http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_jpegs_256.zip.002)
[part3](http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_jpegs_256.zip.003)

- UCF101 Flow: [part1](http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_tvl1_flow.zip.001)
[part2](http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_tvl1_flow.zip.002)
[part3](http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_tvl1_flow.zip.003)

# Use it on your own dataset
- Our [Optical flow extraction tool](https://github.com/feichtenhofer/gpu_flow) provides OpenCV wrappers for optical flow extraction on a GPU.

