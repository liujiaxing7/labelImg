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

IMAGE_SIZE_YOLOV11 = 640
PIXEL_MEAN = [123, 117, 104]
THRESHOLD_YOLOV11 = 0.25

MODEL = CN()
MODEL.CENTER_VARIANCE = 0.1
MODEL.SIZE_VARIANCE = 0.2

from libs.detector.ssd.postprocess.utils.nms import boxes_nms
from libs.detector.utils.utils import TopK
import numpy as np

NMS_THRESHOLD = 0.45
# CONFIDENCE_THRESHOLD = 0.01
MAX_PER_CLASS = -1
MAX_PER_IMAGE = 100
# MAX_PER_IMAGE = 2
BACKGROUND_ID = 0
import time
from libs.detector.ssd.postprocess.utils.nms import boxes_nms

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def iou(box1, boxes):
    x1, y1, x2, y2 = box1
    xx1 = np.maximum(x1, boxes[:, 0])
    yy1 = np.maximum(y1, boxes[:, 1])
    xx2 = np.minimum(x2, boxes[:, 2])
    yy2 = np.minimum(y2, boxes[:, 3])

    w = np.maximum(0, xx2 - xx1)
    h = np.maximum(0, yy2 - yy1)
    inter = w * h

    area1 = (x2 - x1) * (y2 - y1)
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    return inter / (area1 + area2 - inter + 1e-6)

def nms(boxes, scores, iou_thres):
    idxs = scores.argsort()[::-1]
    keep = []
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)
        if len(idxs) == 1:
            break

        ious = iou(boxes[i], boxes[idxs[1:]])
        idxs = idxs[1:][ious < iou_thres]
    return keep

def _make_grid(nx=20, ny=20):
    xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
    return np.stack((xv, yv), 2).reshape((1, 1, ny, nx, 2)).astype(np.float32)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def PostProcessor_YOLOV11Pose(pred, nums):
    """
    自动识别单类/多类 YOLO Pose 输出
    pred shape: [1, feat_dim, N] or [1, 1, feat_dim, N]
    """

    # ----------- 兼容两种导出格式 ----------

    pred = pred[0][0]        # [feat_dim, N]
    feat_dim, N = pred.shape

    # ------------- 固定部分（xywh + obj_conf）-------------
    boxes = pred[:4, :].T           # (N,4)
    obj_conf = pred[4, :]           # (N,)

    # 判断是否含 class_conf
    kpts_dim = 51
    class_dim = feat_dim - (4 + 1 + kpts_dim)

    # ---------------------
    # Case A：单类模型（当前模型）
    # ---------------------
    if nums == 1 or class_dim <= 0:
        total_conf = obj_conf
        cls_id = np.zeros_like(obj_conf, dtype=np.int32)

    # ---------------------
    # Case B：多类模型
    # ---------------------
    else:
        class_conf = pred[5 : 5 + nums, :].T      # (N, nums)

        cls_id = np.argmax(class_conf, axis=1)
        cls_score = class_conf[np.arange(N), cls_id]
        total_conf = obj_conf * cls_score

    # ----------- 置信度过滤 ------------
    keep = total_conf > THRESHOLD_YOLOV11
    boxes = boxes[keep]
    total_conf = total_conf[keep]
    cls_id = cls_id[keep]

    # ----------- xywh → xyxy ------------
    boxes_xyxy = np.zeros_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

    # -------------- NMS ----------------
    keep_idx = nms(boxes_xyxy, total_conf, NMS_THRESHOLD)
    boxes_xyxy = boxes_xyxy[keep_idx]
    total_conf = total_conf[keep_idx]
    cls_id = cls_id[keep_idx]

    final_boxes = np.concatenate(
        (boxes_xyxy, total_conf[:, None], cls_id[:, None]),
        axis=1
    )

    return [final_boxes]


