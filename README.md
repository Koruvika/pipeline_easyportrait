## Convert FPN model from Pytorch to ONNX

```shell
python3 tools/pytorch2onnx.py config local_configs/easy_portrait_experiments/fpn/fpn.resnet50.512x512.easyportrait.20k.py \
    --checkpoint /home/duongdhk/backup/duongdhk/checkpoints/fpn/fpn.resnet50.512x512.pth \
    --input-img /home/duongdhk/backup/duongdhk/datasets/EasyPortrait/images/test/ffd0cc78-260e-4089-a4f1-164fa5c0bdd3.jpg \
    --output-file /home/duongdhk/backup/duongdhk/checkpoints/fpn/fpn.resnet50.512x512.onnx \
    --opset-version 11 \
    --verify \
    --dynamic-export
```

Important arguments to convert pytorch model to onnx format
- config: file config for pre-processing image of FPN
- checkpoint: pytorch checkpoint file
- input-img: image used to assert onnx model
- output-file: onnx save path

more detail about arguments, please see `tools/pytorch2onnx.py`

more detail about pytorch checkpoints, please visit [easyportrait](https://github.com/hukenovs/easyportrait)
