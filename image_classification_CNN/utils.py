#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from sympy import cxxcode

import pyrealsense2 as rs
import cv2
import numpy as np
import time

pipeline = rs.pipeline()  # 定义流程pipeline
config = rs.config()  # 定义配置config
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
profile = pipeline.start(config)  # 流程开始
time.sleep(1.0)
align_to = rs.stream.color  # 与color流对齐
align = rs.align(align_to)



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


if __name__ == '__main__':
    try:
        start_time = time.time()
        i = 1
        times = 11
        while True:
            intr, intr_matrix, depth_intrin, color_image, depth_image, aligned_depth_frame, intr_coeffs = get_aligned_images()  # 获取对齐的图像与相机内参
            if not depth_image.any() or not color_image.any():
                continue
            end_time = time.time()
            if end_time - start_time <= 1.0:
                print(i)
                continue
            else:
                print(end_time - start_time)
                start_time = time.time()
                # print(color_image.shape)
                cv2.imshow('color_image', color_image)
                for j in range(1, 6):
                    tip_image = color_image[127 + i - 3:162 + i - 3, 718 + j - 3:753 + j - 3]
                    cv2.imshow('tip_image', tip_image)
                    index = i + j - 1 + 9 * times
                    cv2.imwrite(f'/home/wsy/image_classification/img_hole/'
                                f'img_hole_{index:02d}.jpg', tip_image)
                i = i + 1
                if i >= 6:
                    cv2.destroyAllWindows()
                    break
                key = cv2.waitKey(1)
                # Press esc or 'q' to close the image window
                if key & 0xFF == ord('q') or key == 27:
                    cv2.destroyAllWindows()
                    break
    finally:
        pipeline.stop()
