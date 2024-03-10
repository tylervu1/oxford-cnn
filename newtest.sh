python tools/test.py \
  --config 'configs/resnet/resnet18_flowers_bs128.py' \
  --checkpoint 'output/resnet18_flowers_bs128/epoch_99.pth' \
  --out 'output/resnet18_flowers_bs128/newtest.json' \
  --metrics accuracy precision recall f1_score

python tools/test.py \
  --config 'configs/resnet/resnet34_b16x8_flowers.py' \
  --checkpoint 'output/resnet34_b16x8_flowers/epoch_99.pth' \
  --out 'output/resnet34_b16x8_flowers/newtest.json' \
  --metrics accuracy precision recall f1_score

python tools/test.py \
  --config 'configs/resnet/resnet50_b16x8_flowers.py' \
  --checkpoint 'output/resnet50_b16x8_flowers/epoch_99.pth' \
  --out 'output/resnet50_b16x8_flowers/newtest.json' \
  --metrics accuracy precision recall f1_score

python tools/test.py \
  --config 'configs/resnet/resnet50_b16x8_flowers_mixup.py' \
  --checkpoint 'output/resnet50_b16x8_flowers_mixup/epoch_99.pth' \
  --out 'output/resnet50_b16x8_flowers_mixup/newtest.json' \
  --metrics accuracy precision recall f1_score
