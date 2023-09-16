import argparse
import csv
import os
import platform
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.torch_utils import select_device, smart_inference_mode

from cadc_loadimage import CADC_LoadIamge
from cadc_detect import detect

class CADC_VISION:
    def __init__(self):
        # self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        # self.cap.set(cv2.CAP_PROP_FPS, 30)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.weights = ROOT / 'yolov5s.pt'
        self.device = select_device("0")
        self.dnn = False
        self.data = ROOT / 'data/coco128.yaml'
        self.half = False  # use FP16 half-precision inference
        self.model = DetectMultiBackend(self.weights, device=self.device, dnn=self.dnn, data=self.data, fp16=self.half)
        self.max_det = 3  # maximum detections per image
        self.line_thickness = 3  # bounding box thickness (pixels)
        self.vid_stride = 1  # video frame-rate stride
        self.conf_thres = 0.25  # confidence threshold
        self.iou_thres = 0.45  # NMS IOU threshold
        self.agnostic_nms = False  # class-agnostic NMS
        self.augment = False  # augmented inference
        self.visualize = False  # visualize features
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size((640, 640), s=self.stride)  # check image size  # inference size (height, width)
        self.im0 = None
        self.bs = 1
        self.model.warmup(imgsz=(1 if self.pt or self.model.triton else self.bs, 3, *self.imgsz))  # warmup

    def shot(self):
        """截图
        """
        # ret, self.im0 = self.cap.read()
        self.im0 = cv2.imread(ROOT / 'data/images/clock6.jpg', 1)

    def run(self):
        # self.im0 = cv2.resize(self.im0, (1920, 1080))
        im = CADC_LoadIamge(im0=self.im0, img_size=self.imgsz, stride=self.stride, auto=self.pt)
        return detect(im0=self.im0, im=im, model=self.model, conf_thres=self.conf_thres, max_det=self.max_det)

if __name__ == "__main__":
    cadc_vision = CADC_VISION()
    cadc_vision.shot()
    result = cadc_vision.run()
    print(result)