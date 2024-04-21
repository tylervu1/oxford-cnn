#!/bin/bash

# how to use bash script
# bash train_models.sh

python tools/train.py \
    --config 'configs/resnet/resnet18_flowers_bs128.py' \
    --work-dir 'output/resnet18_flowers_bs128'

python tools/train.py \
    --config 'configs/resnet/resnet34_b16x8_flowers.py' \
    --work-dir 'output/resnet34_b16x8_flowers'

python tools/train.py \
    --config 'configs/resnet/resnet50_b16x8_flowers.py' \
    --work-dir 'output/resnet50_b16x8_flowers'

python tools/train.py \
    --config 'configs/resnet/resnet50_b16x8_flowers_mixup.py' \
    --work-dir 'output/resnet50_b16x8_flowers_mixup'

python tools/test.py \
    --config 'configs/resnet/resnet34_b16x8_flowers.py' \
    --checkpoint 'output/resnet34_b16x8_flowers/epoch_99.pth' \
    --out 'output/resnet34_b16x8_flowers/test.json'

python tools/test.py \
    --config 'configs/resnet/resnet50_b16x8_flowers.py' \
    --checkpoint 'output/resnet50_b16x8_flowers/epoch_99.pth' \
    --out 'output/resnet50_b16x8_flowers/test.json'

python tools/test.py \
    --config 'configs/resnet/resnet50_b16x8_flowers_mixup.py' \
    --checkpoint 'output/resnet50_b16x8_flowers_mixup/epoch_99.pth' \
    --out 'output/resnet50_b16x8_flowers_mixup/test.json'

################################################################
## commands below run into memory issues on 2080 gpu (fixed?) ##
################################################################

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
   
############################################
## commands below run ternarized Resnet18 ##
############################################

python tools/train.py \
   --config 'configs/resnet/resnet18_flowers_bs128_alltern.py' \
   --work-dir 'output/resnet18_flowers_bs128_alltern'
   
python tools/test.py \
   --config 'configs/resnet/resnet18_flowers_bs128_alltern.py' \
   --checkpoint 'output/resnet18_flowers_bs128_alltern/epoch_99.pth' \
   --out 'output/resnet18_flowers_bs128_alltern/test.json'