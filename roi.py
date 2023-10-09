#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
# @Time : 2023/8/21 09:52:34
# @Author : Taylor
# @File : [Code] --> calc_scale.py
# @Software: PyCharm
# @Function: 线性光照区域的特征提取相关算法
import heapq

import cv2
import numpy as np


def get_line_roi(img_data):
    # 转灰度图像
    gray = cv2.cvtColor(img_data, cv2.COLOR_RGB2GRAY)

    # 图像平滑
    gray = cv2.blur(src=gray, ksize=(5, 5))
    gray = cv2.Sobel(src=gray, ddepth=cv2.CV_8U, dx=0, dy=1, ksize=5)
    cov_kernel = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='float32').T / 11
    gray = cv2.filter2D(gray, -1, kernel=cov_kernel, anchor=(0, 11))
    ret2, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(contours, key=cv2.contourArea, reverse=True)  # 所有轮廓按面积排序
    cnt = cnts[0]  # 第 0 个轮廓，面积最大的轮廓，(664, 1, 2)
    # end_points = get_end_points(cnt)
    # cv2.line(img_data, end_points[0], end_points[1], (0,0,255), 2)
    # print(f"rect: {rect}")
    # scale = calc_scale(end_points, cnt)
    # x, y, w, h = cv2.boundingRect(cnt)
    # brcnt = np.array([[[x, y]], [[x + w, y]], [[x + w, y + h]], [[x, y + h]]])
    # cv2.polylines(img_data, [cnt], True, (0,0,255), 2)  # 绘制拟合轮廓
    # cv2.polylines(img_data, [brcnt], True, (255,0,255), 2)  # 绘制拟合轮廓

    # cv2.drawContours(img_data, [brcnt], -1, (255, 255, 255), 2)
    # 外接凸包计算面积
    # hull = cv2.convexHull(cnt)
    # cv2.polylines(img_data, [hull], True, (0,255,255), 2)  # 绘制拟合轮廓
    # area = cv2.contourArea(hull)

    # print(area)
    # fit_poly = cv2.approxPolyDP(cnt, 3, True)
    # contour_image = cv2.drawContours(image=img_data, contours=cnt, contourIdx=-1, color=(0, 255, 0), thickness=-1)
    # contour_image = cv2.drawContours(image=img_data, contours=contours, contourIdx=max_idx, color=(0, 255, 0), thickness=2)

    # return contour_image
    # res = {
    #     "img": img_data,
    #     "endpoints": end_points,
    #     "scale": scale
    #
    # }
    # print(res)
    # return img_data
    # return res
    # return img_data, area
    return cnt


def calc_scale(end_points, cnt):
    l_point, r_point = end_points[0], end_points[1]
    lx, ly = l_point
    rx, ry = r_point
    # print(444, lx,ly, rx,ry)
    y_end_mean = int((ly + ry) / 2)
    x_end_width = abs(rx - lx)
    y_cnt_mean = int(sum(cnt[:, :, 1]) / len(cnt))
    scale = abs(y_cnt_mean - y_end_mean) * x_end_width
    # print("y_end_mean:", y_end_mean)
    # print("x_end_width:", x_end_width)
    # print("y_cnt_mean:", y_cnt_mean)
    return scale


def get_end_points(roi_points):
    """获取最小外接矩形的上面两个端点"""
    # 获取最小外接矩阵，中心点坐标，宽高，旋转角度 (center(x,y), (width, height), angle of rotation)
    rect = cv2.minAreaRect(roi_points)
    print(rect)
    box = np.intp(cv2.boxPoints(rect)).tolist()  # 获取4个顶点的坐标值
    # end_points = heapq.nsmallest(2, box, key=lambda s: s[1])
    end_points = list(sorted(box, key=lambda x: x[1])[:2])
    print(333, end_points)  # [[81, 311], [1173, 335]]
    return end_points


def get_hull_scale(cnts):
    """外凸包面积，没用上"""
    hull = cv2.convexHull(cnts)
    hull_area = cv2.contourArea(hull)
    return hull_area


if __name__ == '__main__':
    # img_path_test = "./data/111.jpg"
    img_path_test = "./data/002.jpg"
    img_data = cv2.imread(img_path_test)
    img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
    img_data = img_data[296:566, :]
    e1 = cv2.getTickCount()
    # your code execution
    result = get_line_roi(img_data)
    e2 = cv2.getTickCount()
    time = ((e2 - e1) / cv2.getTickFrequency()) * 1000
    cv2.namedWindow('123', cv2.WINDOW_NORMAL)
    cv2.imshow("123", result)
    print("time:", time, "ms")
    # 加入True判断，使得窗口可持续运行
    while True:
        # if cv2.waitKey(30) & 0xFF == ord('q'):
        if cv2.waitKey(30) & 0xFF == 27:
            break

    cv2.destroyAllWindows()
    # cv2.imwrite("321.jpg",img_data)
    print()
    pass
