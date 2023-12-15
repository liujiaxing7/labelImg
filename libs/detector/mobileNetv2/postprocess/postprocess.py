#!/usr/bin/python3 python
# encoding: utf-8
'''
@author: 焦子傲
@contact: jiao1943@qq.com
@file: postprocess.py
@time: 2021/4/9 13:56
@desc:
'''
from yacs.config import CfgNode as CN
from libs.detector.utils.utils import Softmax
import cv2
THRESHOLD_CLASSIFY = 0.5

MODEL = CN()
MODEL.CENTER_VARIANCE = 0.1
MODEL.SIZE_VARIANCE = 0.2

import numpy as np


def _make_grid(nx=20, ny=20):
    xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
    return np.stack((xv, yv), 2).reshape((1, 1, ny, nx, 2)).astype(np.float32)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def PostProcessor_MobileNetV2(predictions):
    classes = np.argmax(predictions)
    scores = np.max(predictions)

    return [classes, scores]