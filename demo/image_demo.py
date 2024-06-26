# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from mmcls.apis import inference_model, init_model, show_result_pyplot


def main():
    parser = ArgumentParser()
    parser.add_argument('--img', default='demo/image_0005.jpg', help='Image file')
    parser.add_argument('--config', default='configs/resnet/resnet18_flowers_bs128.py',help='Config file')
    parser.add_argument('--checkpoint', default='output/resnet18_flowers_bs128/epoch_100.pth')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    # test a single image
    result = inference_model(model, args.img)
    # show the results
    show_result_pyplot(model, args.img, result)
    print(result)


if __name__ == '__main__':
    main()
