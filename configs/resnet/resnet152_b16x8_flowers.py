_base_ = [
    '../_base_/models/resnet152_flowers.py',
    '../_base_/datasets/flowers_bs8.py',
    '../_base_/schedules/flowers_bs16.py', '../_base_/default_runtime.py'
]
