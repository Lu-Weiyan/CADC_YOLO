import cv2
import numpy as np


def center_circle(image_crop=None, image_origin=None):
    if image_crop is not None and image_origin is not None:
        h_crop = image_crop.shape[0]
        w_crop = image_crop.shape[1]
        h_origin = image_origin.shape[0]
        w_origin = image_origin.shape[1]
        # crop_points = np.float32([[0, 0], [0, w_crop], [h_crop, 0]])
        # origin_points = np.float32([[0, 0], [0, w_origin], [h_origin, 0]])
        # affine_matrix = cv2.getAffineTransform(crop_points, origin_points) # crop转origin
        image_edge = cv2.Canny(image_crop, 180, 60, L2gradient=True)
        contours, hierarchy = cv2.findContours(image_edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        area_list = []
        for contour in contours:
            area_list.append(cv2.contourArea(contour))
        if len(area_list) != 0:
            max_area_index = area_list.index(max(area_list))
            (crop_x, crop_y), r = cv2.minEnclosingCircle(contours[max_area_index])
            # print(f"x:{crop_x},y:{crop_y},r:{r}")
            # cv2.circle(image_crop, center=(int(crop_x), int(crop_y)), radius=int(r), color=(0, 0, 0), thickness=10)
            # cv2.drawContours(image_crop, contours, max_area_index, (0, 0, 0))
            # cv2.imshow("", image_crop)
            # cv2.imshow("1", image_edge)
            # cv2.waitKey(0)
            # cv2.destroyWindow("")
            # cv2.destroyWindow("1")
            crop_center_circle = np.array([crop_x, crop_y], dtype=np.float32)
            # # 添加一个额外的维度 [x, y] -> [[x, y]]
            # crop_center_circle = np.expand_dims(crop_center_circle, axis=0)
            # print(crop_center_circle)
            # origin_center_circle = cv2.transform(crop_center_circle, affine_matrix)
            return crop_center_circle, r
        else:
            return None, None

