python3 tools/pytorch2onnx.py config local_configs/easy_portrait_experiments/fpn/fpn.resnet50.512x512.easyportrait.20k.py \
    --checkpoint /home/duongdhk/backup/duongdhk/checkpoints/fpn/fpn.resnet50.512x512.pth \
    --input-img /home/duongdhk/backup/duongdhk/datasets/EasyPortrait/images/test/ffd0cc78-260e-4089-a4f1-164fa5c0bdd3.jpg \
    --output-file /home/duongdhk/backup/duongdhk/checkpoints/fpn/fpn.resnet50.512x512.onnx \
    --opset-version 11 \
    --verify \
    --dynamic-export