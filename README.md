# Midterm Report Progress
## Progress from Tyler Vu + Helen Zhao
### Setup
The initial resnet18 model is the same one from Tutorial 1. We went ahead and implemented this model on the HKU Phase 2
GPU clusters. To set up the environment (assuming this is your first time running this), run the following:
```shell
git clone https://github.com/tylervu1/oxford-cnn
conda create -n mmcls python=3.7 -y
conda activate mmcls
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch
pip install mmcv==1.5.0
pip install mmcv-full==1.5.0
cd oxford-cnn
pip install -e .
pip install yapf==0.40.1
```

### Training + other models
We were also able to successfully implement resnet34_flowers and resnet50_flowers. Training typically takes a long time so to have this run in the background, the bash script train_models.sh was added (basically runs all your training commands consecutively). To use this, run the following:

```shell
chmod +x train_models.sh
./train_models.sh
```

^ also feel free to take a look at this script to see what commands are running to adjust accordingly

We also attempted to implement resnet50_flowers_cutmix, resnet50_flowers_mixup, resnet101_flowers, resnet152_flowers. However, we ran into issues usually regarding memory issues.

### Test 
You can test the trained model by running the following command
```shell
python tools/test.py \
  --config 'configs/resnet/resnet18_flowers_bs128.py' \
  --checkpoint 'output/resnet18_flowers_bs128/epoch_99.pth' \
  --out 'output/resnet18_flowers_bs128/test.json'

python tools/test.py \
  --config 'configs/resnet/resnet34_b16x8_flowers.py' \
  --checkpoint 'output/resnet34_b16x8_flowers/epoch_99.pth' \
  --out 'output/resnet34_b16x8_flowers/test.json'

python tools/test.py \
  --config 'configs/resnet/resnet50_b16x8_flowers.py' \
  --checkpoint 'output/resnet50_b16x8_flowers/epoch_99.pth' \
  --out 'output/resnet50_b16x8_flowers/test.json'

```
The output file will be saved in the ```--out```.

I found that testing results were pretty accurate (97.7% testing classification using the 100th epoch). 200 epochs seemed a big redundant, so running 100 epochs should be fine. However, if we want to see further variations/performance improvements between models, we could potentially lower the epochs even further.

At the time I am writing this readme, I am currently training resnet34_flowers and resnet50_flowers, so I will update the testing results once finished. For our midterm report, it would be good to report the results (hopefully improved) with these larger models. However, either further fine-tuning or other models such as VGG or VIT would be beneficial to build a sufficient midterm report.

## Progress from Robby Chu

### VGG11 and  VGG13
The above two models were successfully implemented. To avoid memory issues, I looked into the ```default_runtime.py``` file and adjusted the interval of saving checkpoints from training models. For the config files, remember to choose the respective runtime file.
```shell
# checkpoint saving
checkpoint_config = dict(interval=<change this number to a higher value>)
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
```

### Test 
You can test the trained model by running the following commands (same format as what Tyler and Helen had done):
```shell
python tools/test.py \
  --config 'configs/vgg/vgg11_b32x8_flowers.py' \
  --checkpoint 'output/vgg11_b32x8_flowers/epoch_99.pth' \
  --out 'output/vgg11_b32x8_flowers/test.json'

python tools/test.py \
  --config 'configs/vgg/vgg13_b32x8_flowers.py' \
  --checkpoint 'output/vgg13_b32x8_flowers/epoch_99.pth' \
  --out 'output/vgg13_b32x8_flowers/test.json'
```
The testing results seem a bit fluctuating and we are still looking for improvements on VGG models.


[![Build Status](https://github.com/open-mmlab/mmclassification/workflows/build/badge.svg)](https://github.com/open-mmlab/mmclassification/actions)
[![Documentation Status](https://readthedocs.org/projects/mmclassification/badge/?version=latest)](https://mmclassification.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/open-mmlab/mmclassification/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmclassification)
[![license](https://img.shields.io/github/license/open-mmlab/mmclassification.svg)](https://github.com/open-mmlab/mmclassification/blob/master/LICENSE)

## Progress from Peter Lu

## Method to run the codes.
Upload the labels.csv file and the images dataset file on to google drive, the the codes will be executable in google colab(Pay special attention to the paths of the uploaded files).


## AlexNet

An AlexNet model was implemented from scratch. The model was designed with 17 classed to fit the usage of flower dataset, also reshaping every image to the standard 227*227 pixels. The 1380 images are divided into 17 folders with each folder containing 80 images from respective class. A CSV file called labels.csv is also created to store a list with images and its labels. The model was trained using V100 in Colab. The training and testing accuracies are plotted against the number of epochs from 1 to 100. The highest testing accuracy is 65.82%, when epoch is 48, not extremely high, but reasonable considering that it has 17 different classes. The code of AlexNet implemented from scratch is in "AlexNet_complete_code.ipynb"

## Introduction

English | [简体中文](/README_zh-CN.md)

MMClassification is an open source image classification toolbox based on PyTorch. It is
a part of the [OpenMMLab](https://openmmlab.com/) project.

Documentation: https://mmclassification.readthedocs.io/en/latest/

![demo](https://user-images.githubusercontent.com/9102141/87268895-3e0d0780-c4fe-11ea-849e-6140b7e0d4de.gif)

## Benchmark and model zoo

Results and models are available in the [model zoo](docs/model_zoo.md).

Supported backbones:

- [x] ResNet
- [x] ResNeXt
- [x] SE-ResNet
- [x] SE-ResNeXt
- [x] RegNet
- [x] ShuffleNetV1
- [x] ShuffleNetV2
- [x] MobileNetV2
- [x] MobileNetV3
- [x] Swin-Transformer

## Citation

```BibTeX
@misc{2020mmclassification,
    title={OpenMMLab's Image Classification Toolbox and Benchmark},
    author={MMClassification Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmclassification}},
    year={2020}
}
```

## Projects in OpenMMLab

- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision.
- [MIM](https://github.com/open-mmlab/mim): MIM Installs OpenMMLab Packages.
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab image classification toolbox and benchmark.
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark.
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab's next-generation platform for general 3D object detection.
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab semantic segmentation toolbox and benchmark.
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab's next-generation action understanding toolbox and benchmark.
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab video perception toolbox and benchmark.
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab pose estimation toolbox and benchmark.
- [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab image and video editing toolbox.
- [MMOCR](https://github.com/open-mmlab/mmocr): OpenMMLab toolbox for text detection, recognition and understanding.
- [MMGeneration](https://github.com/open-mmlab/mmgeneration): OpenMMlab toolkit for generative models.
