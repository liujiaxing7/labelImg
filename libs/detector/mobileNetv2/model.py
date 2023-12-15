#!/usr/bin/python3 python
# encoding: utf-8
'''
@author: 焦子傲
@contact: jiao1943@qq.com
@file: model.py
@time: 2021/4/14 10:56
@desc:
'''
import os
from libs.detector.ssd.onnxmodel import ONNXModel
from libs.detector.yolov3.postprocess.postprocess import load_class_names
from libs.detector.mobileNetv2.preprocess import pre_process
from libs.detector.mobileNetv2.postprocess.postprocess import PostProcessor_MobileNetV2
from libs.detector.mobileNetv2.postprocess.postprocess import THRESHOLD_CLASSIFY
import cv2

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__)).split('libs')[0]


class MobileNetV2(object):
    def __init__(self, file='./config/human/yolov5.onnx', class_sel=[]):
        class_path = os.path.split(file)[0]
        self.classes = load_class_names(class_path + "/classes.names")
        self.class_sel = class_sel

        if os.path.isfile(file):
            self.net = ONNXModel(file)
        else:
            raise IOError("no such file {}".format(file))

    def forward(self, image):
        image = pre_process(image)
        out = self.net.forward(image)
        result = PostProcessor_MobileNetV2(out)

        # TODO : get rect
        shapes = []
        results_box = []
        if result[1] > THRESHOLD_CLASSIFY:
            label = result[0]
            score = result[1]
            y = 175
            y2 = 225
            x = 295
            x2 = 355

            x, y, x2, y2, score, label = int(x), int(y), int(x2), int(y2), float(score), int(label)
            shapes.append((self.classes[label], [(x, y), (x2, y), (x2, y2), (x, y2)], None, None, False, 0))
            results_box.append([x, y, x2, y2, score, self.classes[label]])
        return shapes, results_box
