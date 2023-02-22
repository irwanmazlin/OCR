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
conda create --name ocr python=3.8

# activate your environment before installation or running your scripts 
conda activate ocr


pip install imutils
``` 

for CPU only support:

```bash
# CPU only support (slow)
pip install tensorflow==2.2.0
```

After that, you should install the Object Detection API, which became much easier now after the latest update.
The official installation instructions can be found [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md), but I will add here the instruction to install it as a python package.



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





