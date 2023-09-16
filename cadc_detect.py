import argparse
import csv
import functools
import os
import platform
import sys
from pathlib import Path
import numpy as np
import torch
from cadc_circle import center_circle


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


def compare_list_radius(x: list, y: list):
    return x[2] - y[2]

@smart_inference_mode()
def detect(
    im0: np.array,  # 原图
    im: np.array,  # 新图
    model,  # 模型文件
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=3,  # maximum detections per image
    view_img=True,  # show results
    classes=74,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    line_thickness=3,  # bounding box thickness (pixels)
):
    """ 检测标靶，返回数值与坐标

    Args:
        im0 (np.array): 原图
        im (np.array): 新图
        model: 模型文件
        itv: 图片显示时长
        conf_thres: 置信度阈值

    Returns:
        list (vision_position): 标靶中心坐标、数值列表
    """
    names = model.names
    dt = (Profile(), Profile(), Profile())
    origin_center_circle = None
    r = None

    with dt[0]:
        im = torch.from_numpy(im).to(model.device)  # Tensor
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        # 没有batch_size 时，在前面添加一个轴
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

    # Inference
    with dt[1]:
        pred = model(im, augment=False, visualize=False)

    # NMS
    """
    pred 向前传播的输出
    conf_thres 置信度阈值
    iou_thres iou阈值
    classes 是否只保留特定的类别
    agnostic_nms 进行nms是否也去除不同类别之间的框
    返回值为list[torch.tensor],长度为batch_size
    每一个torch.tensor的shape为(num_boxes, 6),内容为box+conf+cls, box为xyxy(左上右下)
    """
    with dt[2]:
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

    # Process predictions
    """
    对每一张图做处理
    循环次数等于batch_size
    """
    xyr_list = []
    for i, det in enumerate(pred):  # per image
        im0_copy = im0.copy()
        annotator = Annotator(im0, line_width=line_thickness, example=str(names))
        if len(det):
            # Rescale boxes from img_size to im0 size
            # 调整预测框坐标，将resize+pad后的img_size调整回im0的size
            # 此时坐标格式为xyxy
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            # 统计检测到的每一个class的预测框数量
            # for c in det[:, 5].unique():
            #     n = (det[:, 5] == c).sum()  # detections per class

            # Write results
            for *xyxy, conf, cls in reversed(det):  # reversed反转列表顺序
                tlbr = torch.tensor(xyxy).view(1, 4).view(-1).tolist()  # 框的左上右下坐标
                img = im0_copy.copy()
                hei = img.shape[0]
                wid = img.shape[1]

                top = int(tlbr[1]) - 5
                down = int(tlbr[3]) + 5
                left = int(tlbr[0]) - 5
                right = int(tlbr[2]) + 5
                if top < 0 or left < 0 or down > hei or right > wid:  # 舍弃边界图片，确保标靶完整
                    continue
                img_crop = img[top:down, left:right]  # 对原图切片，截取标靶
                crop_center_circle, r = center_circle(img_crop, im0_copy)
                origin_center_circle = np.array([crop_center_circle[0]+left, crop_center_circle[1]+top]).astype(int)
                r = int(r)
                xyr_list.append([origin_center_circle[0], origin_center_circle[1], r])
                print(f"origin_center_circle:{origin_center_circle}, r:{r}")

                # 在原图上画框、圆
                if view_img:  # Add bbox to image
                    c = int(cls)  # integer class
                    label = f'{names[c]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(c, True))
                    cv2.circle(im0, (origin_center_circle[0], origin_center_circle[1]), r, (0, 255, 0), 3)

        im0 = annotator.result()

        xyr_list.sort(key=functools.cmp_to_key(compare_list_radius))
        print(xyr_list)
        origin_center_circle_with_r_min = np.array([xyr_list[0][0], xyr_list[0][1]])
        r_min = xyr_list[0][2]
        # 若设置展示，则画出图片/视频
        if view_img:
            cv2.imshow('0', im0)
            cv2.waitKey(0)
            cv2.destroyWindow('0')

    return origin_center_circle_with_r_min, r_min
