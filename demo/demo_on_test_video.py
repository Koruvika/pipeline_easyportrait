import os.path
from argparse import ArgumentParser
import sys
from glob import glob
import cv2
import numpy as np
from tqdm import tqdm
from retinaface.pre_trained_models import get_model

sys.path.insert(0, '')
from mmseg.apis import inference_segmentor, init_segmentor
from mmseg.core.evaluation import get_palette


def main():
    parser = ArgumentParser()
    parser.add_argument(
        '--test-folder',
        default='/home/duongdhk/backup/duongdhk/datasets/youtube8m_v2'
    )
    parser.add_argument(
        '--config',
        help='Config file',
        default='local_configs/easy_portrait_experiments/fpn/fpn.resnet50.512x512.easyportrait.20k.py',
        # default='local_configs/easy_portrait_experiments/segformer/B2/segformer.b2.512x512.easyportrait.20k.py'
    )
    parser.add_argument(
        '--checkpoint',
        help='Checkpoint file',
        default='/home/duongdhk/backup/duongdhk/checkpoints/fpn/fpn.resnet50.512x512.pth',
        # default='/home/duongdhk/backup/duongdhk/checkpoints/seg-former/b2/segformer.b2.512x512.pth'
    )
    parser.add_argument('--out-folder', default='/home/duongdhk/backup/duongdhk/evaluation/youtube8m_v2', help='Path to output file')
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
    face_detector = get_model("resnet50_2020-07-20", max_size=2048, device="cuda:0")

    s = 0.5
    list_src_fp = glob(args.test_folder + "/**/*.mp4", recursive=True)

    # test a single image
    for src_fp in list_src_fp:
        print(src_fp)
        dst_fp = src_fp.replace(args.test_folder, args.out_folder)
        os.makedirs(os.path.dirname(dst_fp), exist_ok=True)
        cap = cv2.VideoCapture(src_fp)
        assert (cap.isOpened())
        input_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        input_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        input_fps = cap.get(cv2.CAP_PROP_FPS)
        video_length = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        load_bar = tqdm(range(int(video_length)))
        writer = cv2.VideoWriter(dst_fp, cv2.VideoWriter_fourcc(*'mp4v'), input_fps, (int(input_width) * 2, int(input_height)), True)
        try:
            for i in load_bar:
                flag, frame = cap.read()
                if not flag:
                    break
                faces = face_detector.predict_jsons(frame)
                if len(faces) != 1 or len(faces[0]['bbox']) != 4:
                    results = np.hstack((frame, frame))
                else:
                    x0, y0, x1, y1 = faces[0]['bbox']
                    h, w, _ = frame.shape
                    new_h, new_w = y1 - y0, x1 - x0
                    x0 = int(x0 - s * new_w if x0 - s * new_w > 0 else 0)
                    x1 = int(x1 + s * new_w if x0 + s * new_w < w else x1)
                    y0 = int(y0 - s * new_h if y0 - s * new_h > 0 else 0)
                    y1 = int(y1 + s * new_h if y0 + s * new_h < h else y1)
                    face_image = frame[y0:y1, x0:x1]
                    load_bar.set_description(f"Image Size is ({y1 - y0} x {x1 - x0}): ")
                    if y1 - y0 > 0 and x1 - x0 > 0:
                        result = inference_segmentor(model, face_image)
                        full_mask = np.zeros((h, w), dtype=np.int64)
                        # result[0] = np.where(result[0] == 7, 7, 0)
                        full_mask[y0:y1, x0:x1] = result[0]

                        # blend raw image and prediction
                        draw_img = model.show_result(
                            frame,
                            [full_mask],
                            palette=get_palette(args.palette),
                            show=False,
                            opacity=args.opacity)

                        results = np.hstack((frame, draw_img))
                    else:
                        results = np.hstack((frame, frame))
                if writer:
                    writer.write(results)
        finally:
            if writer:
                writer.release()
            cap.release()


if __name__ == '__main__':
    main()
