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
from libs.detector.yolov5.preprocess import preProcessPadding
from libs.detector.yolov5.postprocess.postprocess import PostProcessor_YOLOV5
from libs.detector.yolov5.postprocess.postprocess import IMAGE_SIZE_YOLOV5, THRESHOLD_YOLOV5
import cv2

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__)).split('libs')[0]


class YOLOv5(object):
    def __init__(self, file='./config/human/yolov5.onnx', class_sel=[]):
        class_path = os.path.split(file)[0]
        self.classes = load_class_names(class_path + "/classes.names")
        self.class_sel = class_sel

        if os.path.isfile(file):
            self.net = ONNXModel(file)
        else:
            raise IOError("no such file {}".format(file))

    def forward(self, image):
        oriY = image.shape[0]
        oriX = image.shape[1]
        image = preProcessPadding(image)
        out = self.net.forward(image)
        results_batch = PostProcessor_YOLOV5(out, len(self.classes) + 5)

        # TODO : get rect
        shapes = []
        results_box = []
        for result in results_batch:
            if len(result) > 0:
                result = result.tolist()
                result = [r for r in result if r[4] > THRESHOLD_YOLOV5]
                for r in result:
                    x, y, x2, y2, score, label = r
                    if int(label) > len(self.classes) - 1 or self.classes[int(label)] not in self.class_sel:
                        continue

                    w = IMAGE_SIZE_YOLOV5 / max(oriX, oriY)

                    y = (y - (IMAGE_SIZE_YOLOV5 - w * oriY) / 2) / w + 1
                    y2 = (y2 - (IMAGE_SIZE_YOLOV5 - w * oriY) / 2) / w + 1
                    x = (x - (IMAGE_SIZE_YOLOV5 - w * oriX) / 2) / w + 1
                    x2 = (x2 - (IMAGE_SIZE_YOLOV5 - w * oriX) / 2) / w + 1

                    x, y, x2, y2, score, label = int(x), int(y), int(x2), int(y2), float(score), int(label)
                    shapes.append((self.classes[label], [(x, y), (x2, y), (x2, y2), (x, y2)], None, None, False, 0))
                    results_box.append([x, y, x2, y2, score, self.classes[label]])
        return shapes, results_box
