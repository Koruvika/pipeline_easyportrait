# Copyright (c) OpenMMLab. All rights reserved.
import os
from argparse import ArgumentParser
import sys

import matplotlib.pyplot as plt
from tqdm import tqdm

# from mmcv.cnn.utils.sync_bn import revert_sync_batchnorm

sys.path.insert(0, '')
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot, timing, get_result
from mmseg.core.evaluation import get_palette


def main():
    parser = ArgumentParser()
    parser.add_argument('--test-folder', default='/home/duongdhk/backup/duongdhk/datasets/EasyPortrait/images/test')
    parser.add_argument('--config', help='Config file',
                        default='local_configs/easy_portrait_experiments/fpn/fpn.resnet50.512x512.easyportrait.20k.py')
    parser.add_argument('--checkpoint', help='Checkpoint file',
                        default='/home/duongdhk/backup/duongdhk/checkpoints/fpn/fpn.resnet50.512x512.pth')
    parser.add_argument('--out-folder', default='/home/duongdhk/backup/duongdhk/evaluation/EasyPortrait/fpn512', help='Path to output file')
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

    list_fn = os.listdir(args.test_folder)

    # test a single image
    for fn in tqdm(list_fn):
        src_fp = os.path.join(args.test_folder, fn)
        dst_fp = os.path.join(args.out_folder, fn)
        result = inference_segmentor(model, src_fp)

        get_result(
            model,
            src_fp,
            result,
            get_palette(args.palette),
            opacity=args.opacity,
            out_file=dst_fp)


if __name__ == '__main__':
    main()
