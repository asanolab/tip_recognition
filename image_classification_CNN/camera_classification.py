#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from sympy import cxxcode

import pyrealsense2 as rs
import cv2
import numpy as np
import time
from test import predict
from cv2 import aruco

pipeline = rs.pipeline()  # 定义流程pipeline
config = rs.config()  # 定义配置config
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
profile = pipeline.start(config)  # 流程开始
time.sleep(1.0)
align_to = rs.stream.color  # 与color流对齐
align = rs.align(align_to)
classes_name = ['hole', 'tip']

corner_leftup = [718, 127]
corner_rightdown = [753, 162]


def get_pixel(img):
    return [640, 360]


def pixel2camera_frame(coor_pixel, depth, intr):
    z = depth
    x = (coor_pixel[0] - intr.ppx) * depth / intr.fx
    y = -(coor_pixel[1] - intr.ppy) * depth / intr.fy
    x1 = z
    y1 = -x
    z1 = y
    return [x1 / 1000.0, y1 / 1000.0, z1 / 1000.0]


def get_aligned_images():
    frames = pipeline.wait_for_frames()  # 等待获取图像帧
    aligned_frames = align.process(frames)  # 获取对齐帧
    aligned_depth_frame = aligned_frames.get_depth_frame()  # 获取对齐帧中的depth帧
    color_frame = aligned_frames.get_color_frame()  # 获取对齐帧中的color帧

    ############### 相机参数的获取 #######################
    intr = color_frame.profile.as_video_stream_profile().intrinsics  # 获取相机内参
    depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics  # 获取深度参数（像素坐标系转相机坐标系会用到）

    # 相机内参转换为np.array
    intr_matrix = np.array([
        [intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]
    ])

    depth_image = np.asanyarray(aligned_depth_frame.get_data())  # 深度图（默认16位）
    depth_image_8bit = cv2.convertScaleAbs(depth_image, alpha=0.03)  # 深度图（8位）
    depth_image_3d = np.dstack(
        (depth_image_8bit, depth_image_8bit, depth_image_8bit))  # 3通道深度图
    color_image = np.asanyarray(color_frame.get_data())  # RGB图
    camera_parameters = {'fx': intr.fx, 'fy': intr.fy,
                         'ppx': intr.ppx, 'ppy': intr.ppy,
                         'height': intr.height, 'width': intr.width,
                         'depth_scale': profile.get_device().first_depth_sensor().get_depth_scale()
                         }

    # print(f"intr: {intr}\nintr_matrix:{intr_matrix}\n "
    # f"depth_intrin: {depth_intrin}\n intr.coeffs: {np.array(intr.coeffs)}\n")
    #######################################################
    # 返回相机内参、相机内参的np、深度参数、彩色图、深度图、齐帧中的depth帧、相机畸变系数的np
    return (intr, intr_matrix, depth_intrin, color_image,
            depth_image, aligned_depth_frame, np.array(intr.coeffs))


def detect_aruco(color_img, intr_matrix, intr_coeffs):
    # 获取dictionary，4*4码，指示位50个
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
    # 创建detector parameters
    parameters = aruco.DetectorParameters()
    # 输入rgb，dictionary，相机内参，相机畸变系数
    detector = aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, rejected_img_points = aruco.detectMarkers(color_img,
                                                            aruco_dict, parameters=parameters)
    # rvec是旋转向量，tvec是平移向量
    rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(corners,
                                                               0.045, intr_matrix,
                                                               intr_coeffs)
    # test
    """
    print("rvec:")
    print(rvec)
    print("tvec:")
    print(tvec)
    print("corners:")
    print(corners)
    """
    img_temp = color_img
    if corners is None or ids is None:
        return img_temp, None, None

    # 计算四个点的中心
    centers = []
    # print("corners:", corners)
    for corner in corners:
        for a_corner in corner:
            # corner为[[],[],[],[]]
            # print("corner:", corner)
            centerx = 0
            centery = 0
            for each_corner in a_corner:
                # each_corner为[ , ]
                # print("each corner:", each_corner)
                centerx += each_corner[0]
                centery += each_corner[1]
            centerx /= 4
            centery /= 4
            centerx = int(centerx)
            centery = int(centery)
            centers.append([centerx, centery])
    # print("centers:", centers)
    # 标出aruco码的位置

    real_ids = []
    # ids is [[1] [2] [3]]
    for each_id in ids:
        # each_id is [1]
        for single_each_id in each_id:
            # single_each_id is 1
            real_ids.append(single_each_id)

    if corners is not None and ids is not None:
        aruco.drawDetectedMarkers(color_img, corners)
        for i in range(rvec.shape[0]):
            cv2.drawFrameAxes(color_img,intr_matrix,intr_coeffs,rvec[i,:,:],tvec[i,:,:],0.03)

        # aruco.drawDetectedMarkers(color_img, corners, ids)
        # 画xyz轴
        # cv2.drawFrameAxes(color_img, intr_matrix, intr_coeffs, rvec, tvec, 0.05)
        # cv2.imshow('image', color_img)
        # print(ids)
        # print(centers)
        # print real_ids
        return img_temp, centers, real_ids


if __name__ == '__main__':
    try:
        start_time = time.time()
        i = 1
        max_conf = -1.0
        while True:
            intr, intr_matrix, depth_intrin, color_image, depth_image, aligned_depth_frame, intr_coeffs = get_aligned_images()  # 获取对齐的图像与相机内参
            if not depth_image.any() or not color_image.any():
                continue
            tip_image = color_image[127:162, 718:753]
            end_time = time.time()

            cls, conf, probs = predict(tip_image, '1734448311.8315692/best.pth')

            """
            if end_time - start_time > 1.0:
                cv2.imwrite(f'camera_tip_image/tip_image_{i:02d}.jpg', tip_image)
                print(cls)
                print(conf)
                print(probs)
                start_time = time.time()
                i = i + 1
            """

            # cv2.imwrite('correct_tip.jpg',tip_image)
            color_image_show = np.copy(color_image)

            image_with_frame, marker_centers, ids = detect_aruco(color_image_show, intr_matrix, intr_coeffs)
            #print(marker_centers)
            #print(ids)

            cv2.rectangle(color_image_show, [718, 127], [753, 162], [0, 255, 0], 2)
            if probs[0][0]>probs[0][1]:
                text = (f"Hole: {probs[0][0]:.2f}, "
                        f"Tip: {probs[0][1]:.2f}")
            else:
                text = (f"Tip: {probs[0][1]:.2f}, "
                        f"Hole: {probs[0][0]:.2f}")
            cv2.putText(color_image_show, text, [20, 690], fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.9,
                        color=[0, 255, 0], thickness=2)
            cv2.imshow('color_image', color_image_show)
            cv2.imshow('tip_image', tip_image)
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        pipeline.stop()
