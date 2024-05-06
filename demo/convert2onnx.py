import argparse

import mmcv
import numpy as np
import onnx_tool
import onnxruntime
import onnx

import torch
from mmcv.runner import load_checkpoint
from mmcv.onnx import register_extra_symbolics
from torch import nn
from tqdm import tqdm

from mmseg.apis.inference import LoadImage, show_result_pyplot
from mmseg.datasets.pipelines import Compose
from mmseg.ops import resize
from mmseg.models import build_segmentor


def _convert_batchnorm(module):
    module_output = module
    if isinstance(module, torch.nn.SyncBatchNorm):
        module_output = torch.nn.BatchNorm2d(module.num_features, module.eps,
                                             module.momentum, module.affine,
                                             module.track_running_stats)
        if module.affine:
            module_output.weight.data = module.weight.data.clone().detach()
            module_output.bias.data = module.bias.data.clone().detach()
            # keep requires_grad unchanged
            module_output.weight.requires_grad = module.weight.requires_grad
            module_output.bias.requires_grad = module.bias.requires_grad
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
    for name, child in module.named_children():
        module_output.add_module(name, _convert_batchnorm(child))
    del module
    return module_output


def _prepare_input_img(img_path,
                       test_pipeline,
                       shape=None,
                       rescale_shape=None):
    # build the data pipeline
    if shape is not None:
        test_pipeline[1]['img_scale'] = (shape[1], shape[0])
    test_pipeline[1]['transforms'][0]['keep_ratio'] = False
    test_pipeline = [LoadImage()] + test_pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # prepare data
    data = dict(img=img_path)
    data = test_pipeline(data)
    imgs = data['img']
    img_metas = [i.data for i in data['img_metas']]

    if rescale_shape is not None:
        for img_meta in img_metas:
            img_meta['ori_shape'] = tuple(rescale_shape) + (3, )

    mm_inputs = {'imgs': imgs, 'img_metas': img_metas}

    return mm_inputs


def _update_input_img(img_list, img_meta_list, update_ori_shape=False):
    # update img and its meta list
    N, C, H, W = img_list[0].shape
    img_meta = img_meta_list[0][0]
    img_shape = (H, W, C)
    if update_ori_shape:
        ori_shape = img_shape
    else:
        ori_shape = img_meta['ori_shape']
    pad_shape = img_shape
    new_img_meta_list = [[{
        'img_shape':
        img_shape,
        'ori_shape':
        ori_shape,
        'pad_shape':
        pad_shape,
        'filename':
        img_meta['filename'],
        'scale_factor':
        (img_shape[1] / ori_shape[1], img_shape[0] / ori_shape[0]) * 2,
        'flip':
        False,
    } for _ in range(N)]]

    return img_list, new_img_meta_list


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default='local_configs/easy_portrait_experiments/fpn/fpn.resnet50.512x512.easyportrait.20k.py',
        # default='local_configs/easy_portrait_experiments/segformer/B2/segformer.b2.512x512.easyportrait.20k.py',
        help='test config file path')
    parser.add_argument(
        '--checkpoint',
        default='/home/duongdhk/backup/duongdhk/checkpoints/fpn/fpn.resnet50.512x512.pth',
        # default='/home/duongdhk/backup/duongdhk/checkpoints/seg-former/b2/segformer.b2.512x512.pth',
        help='checkpoint file')
    parser.add_argument(
        '--output-file', type=str,
        default='/home/duongdhk/backup/duongdhk/checkpoints/fpn/fpn.resnet50.512x512.uint8.onnx',
        # default='/home/duongdhk/backup/duongdhk/checkpoints/seg-former/b2/segformer.b2.512x512.onnx',
    )
    parser.add_argument(
        '--input-img', type=str, help='Images for input',
        default='/home/duongdhk/backup/duongdhk/datasets/EasyPortrait/images/test/ffd0cc78-260e-4089-a4f1-164fa5c0bdd3.jpg')

    parser.add_argument(
        '--time',
        action='store_false'
    )
    parser.add_argument(
        '--show',
        action='store_false'
    )
    parser.add_argument(
        '--check',
        action='store_false'
    )

    return parser.parse_args()


def run_export(model, inputs, outfile, show, check, eval_time, opset_version=11, verbose=True, gpu=True):
    if gpu:
        model.cuda().eval()
    else:
        model.eval()

    if isinstance(model.decode_head, nn.ModuleList):
        num_classes = model.decode_head[-1].num_classes
    else:
        num_classes = model.decode_head.num_classes

    imgs = inputs.pop('imgs')
    img_metas = inputs.pop('img_metas')
    if gpu:
        img_list = [img[None, :].cuda() for img in imgs]
    else:
        img_list = [img[None, :] for img in imgs]
    img_meta_list = [[img_meta] for img_meta in img_metas]
    img_list, img_meta_list = _update_input_img(img_list, img_meta_list, True)

    register_extra_symbolics(opset_version)
    with torch.no_grad():
        torch.onnx.export(
            model,
            (img_list, img_meta_list, False, dict(rescale=True)),
            outfile,
            input_names=['input'],
            output_names=['output'],
            export_params=True,
            keep_initializers_as_inputs=False,
            verbose=verbose,
            opset_version=opset_version,
            dynamic_axes=None
        )
        onnx_tool.model_profile(outfile, verbose=False)
        print(f'Successfully exported ONNX model: {outfile}')

    ### setup onnx model
    onnx_model = onnx.load(outfile)
    onnx.checker.check_model(onnx_model)
    sess = onnxruntime.InferenceSession(outfile, providers=["CUDAExecutionProvider"])

    ### setup input for onnx model
    input_all = [node.name for node in onnx_model.graph.input]
    input_initializer = [node.name for node in onnx_model.graph.initializer]
    net_feed_input = list(set(input_all) - set(input_initializer))
    inputs = img_list[0].detach().cpu().numpy()

    # get results
    pytorch_result = np.stack(model(img_list, img_meta_list, return_loss=False), 0)
    onnx_result = sess.run(None, {net_feed_input[0]: inputs})[0][0]

    if eval_time:
        ## Pytorch time
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        repetitions = 300
        timings=np.zeros((repetitions,1))
        load_bar = tqdm(range(repetitions))

        with torch.no_grad():
            for rep in range(10):
                _ = model(img_list, img_meta_list, return_loss=False)

        with torch.no_grad():
            for rep in load_bar:
                starter.record()
                _ = model(img_list, img_meta_list, return_loss=False)
                ender.record()
                # WAIT FOR GPU SYNC
                torch.cuda.synchronize()
                timings[rep] = starter.elapsed_time(ender)
                load_bar.set_description(f"Pytorch Time: {timings[rep]}")

        ## ONNX time
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        repetitions = 300
        timings=np.zeros((repetitions,1))
        load_bar = tqdm(range(repetitions))

        for rep in range(10):
            _ = sess.run(None, {net_feed_input[0]: inputs})[0][0]

        for rep in load_bar:
            starter.record()
            _ = sess.run(None, {net_feed_input[0]: inputs})[0][0]
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            timings[rep] = starter.elapsed_time(ender)
            load_bar.set_description(f"ONNX Time: {timings[rep]}")

    if show:
        import os.path as osp

        import cv2
        img = img_meta_list[0][0]['filename']
        if not osp.exists(img):
            img = imgs[0][:3, ...].permute(1, 2, 0) * 255
            img = img.detach().numpy().astype(np.uint8)
            ori_shape = img.shape[:2]
        else:
            ori_shape = LoadImage()({'img': img})['ori_shape']

        # resize onnx_result to ori_shape
        onnx_result_ = cv2.resize(onnx_result[0].astype(np.uint8),
                                  (ori_shape[1], ori_shape[0]))
        show_result_pyplot(
            model,
            img, (onnx_result_, ),
            palette=model.PALETTE,
            block=False,
            title='ONNXRuntime',
            opacity=0.5)

        # resize pytorch_result to ori_shape
        pytorch_result_ = cv2.resize(pytorch_result[0].astype(np.uint8),
                                     (ori_shape[1], ori_shape[0]))
        show_result_pyplot(
            model,
            img, (pytorch_result_, ),
            title='PyTorch',
            palette=model.PALETTE,
            opacity=0.5)

    if check:
        onnx_result_ = cv2.resize(onnx_result[0].astype(np.uint8),
                                  (ori_shape[1], ori_shape[0]))
        pytorch_result_ = cv2.resize(pytorch_result[0].astype(np.uint8),
                                     (ori_shape[1], ori_shape[0]))
        # compare results
        np.testing.assert_allclose(
            pytorch_result_.astype(np.float32) / num_classes,
            onnx_result_.astype(np.float32) / num_classes,
            rtol=1e-5,
            atol=1e-5,
            err_msg='The outputs are different between Pytorch and ONNX')
        print('The outputs are same between Pytorch and ONNX')


def main():
    args = parse_args()

    # prepare config and input
    cfg = mmcv.Config.fromfile(args.config)
    cfg.model.pretrained = None
    input_shape = (1, 3, 512, 512)
    cfg.model.train_cfg = None

    # prepare segmentation model
    segmentor = build_segmentor(
        cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
    # convert SyncBN to BN
    segmentor = _convert_batchnorm(segmentor)

    # load checkpoint
    if args.checkpoint:
        checkpoint = load_checkpoint(
            segmentor, args.checkpoint, map_location='cuda:0')
        segmentor.CLASSES = checkpoint['meta']['CLASSES']
        segmentor.PALETTE = checkpoint['meta']['PALETTE']


    # read input
    preprocess_shape = (input_shape[2], input_shape[3])
    mm_inputs = _prepare_input_img(
        args.input_img,
        cfg.data.test.pipeline,
        shape=preprocess_shape,
        rescale_shape=None)

    run_export(
        segmentor,
        mm_inputs,
        args.output_file,
        args.show,
        args.check,
        args.time
    )



if __name__ == "__main__":
    main()