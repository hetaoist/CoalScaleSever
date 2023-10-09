#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
# @Time : 2023/8/21 19:55:15
# @Author : Taylor
# @File : [Code] --> deal.py
# @Software: PyCharm
# @Function: 写处理逻辑保证测量的准确性和稳定性
import time

import cv2
import numpy as np

import gol
from roi import get_line_roi, get_end_points, calc_scale
from pylogger import log


def pic_preprocess(img_data):
    """图像还没有被裁剪ROI区域前的处理函数"""
    cnt = get_line_roi(img_data)
    end_points = get_end_points(cnt)
    scale = calc_scale(end_points, cnt)
    x, y, w, h = cv2.boundingRect(cnt)
    valid = False
    # 筛选有效模板图像信息
    if scale > 10000 and w / h > 5:
        gol.get_value("end_points_list").append(end_points)
        gol.get_value("scale_list").append(scale)
        gol.get_value("position_list").append((x, y, w, h))
        valid = True
    return valid


def pic_process(img_data):
    """图像还没有被裁剪ROI区域前的处理函数"""
    now = time.time()
    x, y, w, h = gol.get_value("position")
    img_data = img_data[y:y + h+20, x:x + w]
    cnt = get_line_roi(img_data)
    model_end_points = gol.get_value("end_points")
    model_end_points = calc_model_points([x, y], model_end_points)
    scale = calc_scale(model_end_points, cnt)
    model_scale = gol.get_value("scale")
    cv2.line(img_data, model_end_points[0], model_end_points[1], (0, 0, 255), 2)
    cv2.polylines(img_data, [cnt], True, (0, 0, 255), 2)  # 绘制拟合轮廓
    rate = round(100 * (model_scale - scale) / model_scale, 2)
    rate = 0 if rate < 0 else rate
    # print(model_scale, scale, "rate:", rate, "time:{} ms".format(str(int(time.time()-now)*1000)))
    # cv2.putText(图像, 要写入的内容, (文字坐标), 字体, 字号, (字体颜色), 字体粗细)
    tm = int((time.time()-now)*1000)
    print("{:<8}".format(scale), "rate:{:>7}%  ".format(rate), "time:{:>4} ms".format(tm))
    cv2.putText(img_data, 'rt:{:>5}% tm:{:>2}ms'.format(rate, tm), (5, h - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (55, 255, 155), 1)
    return img_data


def calc_model_points(init_point, end_points):
    l_point, r_point = end_points[0], end_points[1]
    lx, ly = l_point
    rx, ry = r_point
    return [[lx - init_point[0], ly - init_point[1]], [rx - init_point[0], ry - init_point[1]]]


def calc_mean_value():
    end_points = np.array(gol.get_value("end_points_list")).mean(axis=0).astype(int)
    scale = np.array(gol.get_value("scale_list")).mean(axis=0).astype(int)
    position = np.array(gol.get_value("position_list")).mean(axis=0).astype(int)
    gol.set_value("end_points", end_points)
    gol.set_value("scale", scale)
    gol.set_value("position", position)


def video_deal(debug=True):
    camera_id = './data/cut_normal.avi' if debug else 0
    cap = cv2.VideoCapture(camera_id)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    gol.set_value("width", frame_width)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    gol.set_value("height", frame_height)
    fps = cap.get(cv2.CAP_PROP_FPS)
    gol.set_value("fps", fps)

    if not cap.isOpened():
        log.logger("Error opening video stream or file")
    valid_count = 1
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if gol.get_value("normal"):
                # 正常处理-计算面积、比例、画图，数据写到
                roi_data = pic_process(frame)
                # 后续可在此处根据需求保存或者显示图片
                # cv2.imshow('Frame', roi_data)
                cv2.imwrite(f'{valid_count}.jpg', roi_data)
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break
            else:
                # 模版预处理--计算端点以及切图, 数值存入全局变量
                frame_valid = pic_preprocess(frame)
                if frame_valid:
                    valid_count += 1
                    # 3s有效时长的模版预处理
                    if valid_count > 180:
                        calc_mean_value()
                        gol.set_value("normal", True)
                        # break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

    pass


if __name__ == '__main__':
    camera_id_test = './data/cut_normal.avi'
    # camera_id_test = './data/cut_all.avi'
    # camera_id_test = './data/001.avi'

    print()
    pass
