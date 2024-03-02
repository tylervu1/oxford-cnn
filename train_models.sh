#!/bin/bash

# how to use bash script
# chmod +x train_models.sh
# ./train_models.sh

# python tools/train.py \
#   --config 'configs/resnet/resnet18_flowers_bs128.py' \
#   --work-dir 'output/resnet18_flowers_bs128'

python tools/train.py \
  --config 'configs/resnet/resnet34_b16x8_flowers.py' \
  --work-dir 'output/resnet34_b16x8_flowers'

python tools/train.py \
  --config 'configs/resnet/resnet50_b16x8_flowers.py' \
  --work-dir 'output/resnet50_b16x8_flowers'

python tools/train.py \
  --config 'configs/resnet/resnet50_b16x8_flowers_mixup.py' \
  --work-dir 'output/resnet50_b16x8_flowers_mixup'

#######################################################
## commands below run into memory issues on 2080 gpu ##
#######################################################

# python tools/train.py \
#   --config 'configs/resnet/resnet101_b16x8_flowers.py' \
#   --work-dir 'output/resnet101_b16x8_flowers'

# python tools/train.py \
#   --config 'configs/resnet/resnet152_b16x8_flowers.py' \
#   --work-dir 'output/resnet152_b16x8_flowers'
