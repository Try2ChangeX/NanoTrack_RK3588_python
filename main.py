from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import time
import argparse

import cv2
import numpy as np
from glob import glob

import sys

sys.path.append(os.getcwd())
from core.config import cfg

from models.rknnlite_rk3588_tracker import NnoTracker_RKNNLite

parser = argparse.ArgumentParser(description='tracking demo')

parser.add_argument('--config', default='./models/config/config.yaml', type=str, help='config file')

parser.add_argument('--save', action='store_true', help='whether visualzie result')

args = parser.parse_args()

def get_frames(video_name):
    if not video_name:
        cap = cv2.VideoCapture(0)
        # warmup
        for i in range(5):
            cap.read()
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break

    elif video_name.endswith('avi') or \
            video_name.endswith('mp4') or \
            video_name.endswith('mov'):
        cap = cv2.VideoCapture(video_name)

        # warmup
        for i in range(50):
            cap.read()

        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    else:
        images = glob(os.path.join(video_name, '*.jp*'))
        images = sorted(images,
                        key=lambda x: int(x.split('/')[-1].split('.')[0]))
        for img in images:
            frame = cv2.imread(img)
            yield frame


def main():
    # load config
    cfg.merge_from_file(args.config)

    # load_weight
    Tback_weight = './weights/track_backbone_T.rknn'
    Xback_weight = './weights/track_backbone_X.rknn'
    Head_weight = './weights/head.rknn'

    video_name = './data/{your_video}'
    tracker = NnoTracker_RKNNLite(Tback_weight, Xback_weight, Head_weight)
    first_frame = True

    # img_savedir = './data/debug_img/'
    # count = 0

    for frame in get_frames(video_name):
        if first_frame:
            # build video writer

            init_rect = [280, 472, 70, 47]
            tracker.init(frame, init_rect)
            first_frame = False
        else:
            t1 = time.time()
            outputs = tracker.track(frame)
            print('fps：', 1. / (time.time() - t1))
            if 'polygon' in outputs:
                polygon = np.array(outputs['polygon']).astype(np.int32)
                cv2.polylines(frame, [polygon.reshape((-1, 1, 2))],
                              True, (0, 255, 0), 3)
                mask = ((outputs['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255)
                mask = mask.astype(np.uint8)
                mask = np.stack([mask, mask * 255, mask]).transpose(1, 2, 0)
                frame = cv2.addWeighted(frame, 0.77, mask, 0.23, -1)
            else:
                bbox = list(map(int, outputs['bbox']))
                cv2.rectangle(frame, (bbox[0], bbox[1]),
                              (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                              (0, 255, 0), 3)
            # cv2.imshow(video_name, frame)
            # cv2.waitKey(30)
            # cv2.imwrite(os.path.join(img_savedir, '%03d.jpg'%count), frame)
            # count += 1


if __name__ == '__main__':
    main()
