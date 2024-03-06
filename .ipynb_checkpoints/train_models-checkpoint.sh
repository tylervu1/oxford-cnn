#!/bin/bash

# how to use bash script
# chmod +x train_models.sh
# ./train_models.sh

# python tools/train.py \
#   --config 'configs/resnet/resnet18_flowers_bs128.py' \
#   --work-dir 'output/resnet18_flowers_bs128'

#python tools/train.py \
#  --config 'configs/resnet/resnet34_b16x8_flowers.py' \
#  --work-dir 'output/resnet34_b16x8_flowers'

#python tools/train.py \
#  --config 'configs/resnet/resnet50_b16x8_flowers.py' \
#  --work-dir 'output/resnet50_b16x8_flowers'

#python tools/test.py \
#  --config 'configs/resnet/resnet34_b16x8_flowers.py' \
#  --checkpoint 'output/resnet34_b16x8_flowers/epoch_99.pth' \
#  --out 'output/resnet34_b16x8_flowers/test.json'

#python tools/test.py \
#  --config 'configs/resnet/resnet50_b16x8_flowers' \
#  --checkpoint 'output/resnet50_b16x8_flowers/epoch_99.pth' \
#  --out 'output/resnet50_b16x8_flowers/test.json'

#python tools/train.py \
#  --config 'configs/vgg/vgg11_b32x8_flowers.py' \
#  --work-dir 'output/vgg11_b32x8_flowers'

#python tools/train.py \
#  --config 'configs/vgg/vgg13_b32x8_flowers.py' \
#  --work-dir 'output/vgg13_b32x8_flowers'

#python tools/train.py \
#  --config 'configs/vgg/vgg13bn_b32x8_flowers.py' \
#  --work-dir 'output/vgg13bn_b32x8_flowers'

#python tools/test.py \
#  --config 'configs/vgg/vgg11_b32x8_flowers.py' \
#  --checkpoint 'output/vgg11_b32x8_flowers/epoch_99.pth' \
#  --out 'output/vgg11_b32x8_flowers/test.json'

#python tools/test.py \
#  --config 'configs/vgg/vgg13_b32x8_flowers.py' \
#  --checkpoint 'output/vgg13_b32x8_flowers/epoch_99.pth' \
#  --out 'output/vgg13_b32x8_flowers/test.json'

#python tools/test.py \
#  --config 'configs/vgg/vgg13bn_b32x8_flowers.py' \
#  --checkpoint 'output/vgg13bn_b32x8_flowers/epoch_99.pth' \
#  --out 'output/vgg13bn_b32x8_flowers/test.json'

# python tools/train.py \
#   --config 'configs/resnet/resnet50_b16x8_flowers_mixup.py' \
#   --work-dir 'output/resnet50_b16x8_flowers_mixup'

#######################################################
## commands below run into memory issues on 2080 gpu ##
#######################################################

python tools/train.py \
   --config 'configs/resnet/resnet101_b16x8_flowers.py' \
   --work-dir 'output/resnet101_b16x8_flowers'

python tools/train.py \
   --config 'configs/resnet/resnet152_b16x8_flowers.py' \
   --work-dir 'output/resnet152_b16x8_flowers'

python tools/test.py \
   --config 'configs/resnet/resnet101_b16x8_flowers.py' \
   --checkpoint 'output/resnet101_b16x8_flowers/epoch_99.pth' \
   --out 'output/resnet101_b16x8_flowers/test.json'

python tools/test.py \
   --config 'configs/resnet/resnet152_b16x8_flowers.py' \
   --checkpoint 'output/resnet152_b16x8_flowers/epoch_99.pth' \
   --out 'output/resnet152_b16x8_flowers/test.json'
