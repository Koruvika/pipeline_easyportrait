# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
import sys

import matplotlib.pyplot as plt

# from mmcv.cnn.utils.sync_bn import revert_sync_batchnorm

sys.path.insert(0, '')
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot, timing, get_result
from mmseg.core.evaluation import get_palette


def main():
    parser = ArgumentParser()
    parser.add_argument('--img', help='Image file',
                        default='/home/duongdhk/backup/duongdhk/datasets/EasyPortrait/images/val/fd29e795-a8db-42d1-8e0d-062ba8d81ebd.jpg')
    parser.add_argument('--config', help='Config file',
                        default='local_configs/easy_portrait_experiments/fpn/fpn.resnet50.512x512.easyportrait.20k.py')
    # parser.add_argument('--config', help='Config file',
    #                     default='/home/duongdhk/projects/research/easyportrait/pipelines/local_configs/easyportrait_experiments_v2/fpn-fp/fpn-fp.py')
    parser.add_argument('--checkpoint', help='Checkpoint file',
                        default='/home/duongdhk/backup/duongdhk/checkpoints/fpn/fpn.resnet50.512x512.pth')
    parser.add_argument('--out-file', default='result.jpg', help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='easy_portrait',
        help='Color palette used for segmentation map')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, args.checkpoint, device=args.device)
    # if args.device == 'cpu':
    #     model = revert_sync_batchnorm(model)
    # test a single image
    result = timing(model, args.img)
    # show the results
    img_results = get_result(
        model,
        args.img,
        result,
        get_palette(args.palette),
        opacity=args.opacity)



if __name__ == '__main__':
    main()
