_base_ = [
    '../_base_/models/vgg11_flowers.py',
    '../_base_/datasets/flowers_bs32.py',
    '../_base_/schedules/flowers_bs16.py',
    '../_base_/default_runtime_interval33.py',
]
optimizer = dict(lr=0.01)
