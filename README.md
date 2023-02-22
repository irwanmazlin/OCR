# Scene Text Detection using East Model



## Introduction





## Roadmap

This tutorial will take you from installation, to running pre-trained detection model, t

1. [Installation](#installation)
2. [Inference with pre-trained models](#inference-with-pre-trained-models)


## Installation

The examples in this repo is tested with python 3.6 and Tensorflow 2.2.0, but it is expected to work with other Tensorflow 2.x versions with python version 3.5 or higher.

It is recommended to install [anaconda](https://www.anaconda.com/products/individual) and create new [environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) for your projects that use the installed packages:

```bash
# create new environment
conda create --name py36-tf2 python=3.6

# activate your environment before installation or running your scripts 
conda activate py36-tf2
``` 

You need first to install tensorflow 2, either with GPU or CPU only support (slow). For Installation with GPU support, you need to have CUDA 10.1 with CUDNN 7.6 to use Tensorflow 2.2.0.
You can check the compatible versions of any tensorflow version with cuda and cudnn versions from [here](https://www.tensorflow.org/install/source#tested_build_configurations).
If you have the matching CUDA and CUDNN versions in your system, install tensorflow-gpu as follows: 

```bash
# if you have NVIDIA GPU with cuda 10.1 and cudnn 7.6
pip install tensorflow-gpu==2.2.0
```

Otherwise, a great feature of Anaconda is that it can automatically install a local version of cudatoolkit that is compatible with your tensorflow version (But you should have the proper nvidia gpu drivers installed).
But be aware that this cuda version will be only available to your python environment (or virtual environment if you created one), and will not be available for other python versions or other environments.

```bash
# installation from anaconda along with cudatoolkit (tf2 version 2.2.0)
conda install -c anaconda tensorflow-gpu==2.2.0

# or to install latest version of tensorflow, just type
conda install -c anaconda tensorflow-gpu
```

for CPU only support:

```bash
# CPU only support (slow)
pip install tensorflow==2.2.0
```

After that, you should install the Object Detection API, which became much easier now after the latest update.
The official installation instructions can be found [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md), but I will add here the instruction to install it as a python package.

Clone the TensorFlow models repository:

```bash
git clone https://github.com/tensorflow/models.git
```

Make sure you have [protobuf compiler](https://grpc.io/docs/protoc-installation/#install-using-a-package-manager) version >= 3.0, by typing `protoc --version`, or install it on Ubuntu by typing `apt install protobuf-compiler`.

Then proceed to the python package installation as follows:

```bash
# remember to activate your python environment first
cd models/research
# compile protos:
protoc object_detection/protos/*.proto --python_out=.
# Install TensorFlow Object Detection API as a python package:
cp object_detection/packages/tf2/setup.py .
python -m pip install .
```

The previous commands installs the object detection api as a python package that will be available in your python environment (or virtual environment if you created one),
and will automatically install all required dependencies if not already installed.

Finally, to test that your installation is correct, type the following command: 

```bash
# Test the installation.
python object_detection/builders/model_builder_tf2_test.py
```

For more installation options, please refer to the original [installation guide](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md).

To run the examples in this repo, you will need some additional dependencies:

```bash
# install OpenCV python package
pip install opencv-python
pip install opencv-contrib-python
```
------------------------------------------------------------

## Inference with pre-trained models


```bash

cd models/
# download the model
wget 


In the tensorflow object detection repo, they provide a tutorial for inference in this [notebook](https://github.com/tensorflow/models/blob/master/research/object_detection/colab_tutorials/inference_from_saved_model_tf2_colab.ipynb), but it is not so clean and needs many improvements.
Therefore, I have created a class for object detection inference [detector.py](detector.py), along with an example script [detect_objects.py](detect_objects.py) to use this class to run the inference with input images, or from a video.   

I encourage you to have a look at the [class file](detector.py) and the [example script](detect_objects.py) and adapt it to your application. But let's first see how to use it to get the inference running with the EfficientDet D0 model we have just downloaded.  

We can provide some argument when running the [detect_objects.py](detect_objects.py) script. For instance, the `--model_path` argument sets the path of the trained model, and the `--path_to_labelmap` points to the labelmap file of your dataset (here we use the one for coco dataset).
To run the detection with set of images, provide a path to the folder containing the images in the argument `--images_dir`.

```bash
python detect_objects.py --model_path models/efficientdet_d0_coco17_tpu-32/saved_model --path_to_labelmap models/mscoco_label_map.pbtxt --images_dir data/samples/images/
```

Sample output from the detection with the pretrained EfficientDet D0 model:

![detcetion-output1](data/samples/output/1.jpg)


You can also select a set of classes to be detected by passing their labels to the argument `--class_ids` as a string with the "," delimiter. For example, using `--class_ids "1,3" ` will do detection for the classes "person" and "car" only as they have id 1 and 3 respectively
(you can check the id and labels from the [coco labelmap](models/mscoco_label_map.pbtxt)). Not using this argument will lead to detecting all objects in the provided labelmap.

Let's use video input by enabling the flag `--video_input`, in addition to detecting only people by passing id 1 to the `--class_ids` argument. The video used for testing is downloaded from [here](https://www.youtube.com/watch?v=pk96gqasGBQ).

```bash
python detect_objects.py --video_input --class_ids "1" --threshold 0.3  --video_path data/samples/pedestrian_test.mp4 --model_path models/efficientdet_d0_coco17_tpu-32/saved_model --path_to_labelmap models/mscoco_label_map.pbtxt
```

------------------------------------------------------------





