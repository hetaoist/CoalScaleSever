#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
# @Time : 2023/8/19 09:52:34
# @Author : Hetao
# @File : [Code] --> calc_scale.py
# @Software: PyCharm
# @Function: TODO
import configparser

import cv2
import numpy as np

Config = configparser.ConfigParser()
Config.read("./config.ini", encoding="utf-8")


def get_calc_scale(img_data):
    # 转灰度图像
    gray = cv2.cvtColor(img_data, cv2.COLOR_RGB2GRAY)

    # 图像平滑
    # gray = cv2.medianBlur(src=gray, ksize=7)
    gray = cv2.blur(src=gray, ksize=(17, 1))
    # gray = cv2.GaussianBlur(src=gray, ksize=(11, 5),sigmaX=3, sigmaY=0)
    # gray = cv2.bilateralFilter(gray, 7, 10, 2)
    # gray = cv2.boxFilter(src=gray, ddepth=-1, ksize=(5, 5), normalize=1)  # normalize=1表示做归一化处理

    # 图像增强
    # gray = cv2.equalizeHist(gray)
    # cv2.normalize(gray, gray, 0, 255, cv2.NORM_MINMAX)
    # 创建ClAHE对象
    # clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(4, 4))
    # 限制对比度的自适应阈值均衡化
    # gray = clahe.apply(gray)

    # 图像锐化
    gray = cv2.Sobel(src=gray, ddepth=cv2.CV_8U, dx=0, dy=1, ksize=5)
    # grayx = cv2.Sobel(src=gray, ddepth=cv2.CV_8U, dx=1, dy=0, ksize=5)
    # grayy = cv2.Sobel(src=gray, ddepth=cv2.CV_8U, dx=0, dy=1, ksize=5)
    # absX  = cv2.convertScaleAbs(grayx)  # ksize 指定 number ✖ number
    # absY  = cv2.convertScaleAbs(grayy)  # ksize 指定 number ✖ number
    # gray = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    # gray = cv2.equalizeHist(gray)

    # 进行Canny边缘检测
    # gray = cv2.Canny(gray, 0, 120)
    # gray = cv2.Laplacian(src=gray, ddepth=-1, ksize=3)  # ddept=-1表示和原图像一样的深度，可调节
    # gray = cv2.Scharr(gray, cv2.CV_8U, 1, 0)

    # 阈值分割
    ret2, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # binary = cv2.adaptiveThreshold(gray, 50, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 2)

    # 形态学处理
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    binary = cv2.dilate(binary, kernel, iterations=1)
    # binary = cv2.erode(binary, kernel, iterations=1)
    # binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    # binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # font = cv2.FONT_HERSHEY_SIMPLEX
    # cv2.putText(gray, 'OpenCV', (10, 500), font, 4, (255, 255, 255), 2)
    # ball=img[280:340,330:390]

    # 提取轮廓
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # OpenCV4~
    cnts = sorted(contours, key=cv2.contourArea, reverse=True)  # 所有轮廓按面积排序
    cnt = cnts[0]  # 第 0 个轮廓，面积最大的轮廓，(664, 1, 2)
    polyFit = cv2.approxPolyDP(cnt, 5, True)
    print("polyFit:", polyFit)
    fitContour = np.zeros(gray.shape[:2], np.uint8)  # 初始化最大轮廓图像
    cv2.polylines(fitContour, [cnt], True, 205, thickness=2)  # 绘制最大轮廓，多边形曲线
    cv2.polylines(fitContour, [polyFit], True, 255, 3) # 绘制拟合轮廓

    # 找到最大轮廓
    area = []
    for k in range(len(contours)):
        area.append(cv2.contourArea(contours[k]))
    max_idx = np.argmax(np.array(area))  # 用于返回一个numpy数组中最大值的索引值
    largestcontour = contours[max_idx]

    # 绘制轮廓
    # contour_image = cv2.drawContours(image=img_data, contours=largestcontour, contourIdx=-1, color=(0, 255, 0), thickness=2)
    contour_image = cv2.drawContours(image=img_data, contours=contours, contourIdx=max_idx, color=(0, 255, 0), thickness=2)

    # 截图只输出截图内容
    x, y, w, h = cv2.boundingRect(largestcontour)
    print(x, y, w, h) # 122 366 1048 170
    # crop = contour_image[y:y + h, x:x + w]
    crop = contour_image[y - 50:y + h + 50, :]  # [296:506, :]

    # 对二值图像进行细化处理
    # thinned = cv2.ximgproc.thinning(binary, cv2.ximgproc.THINNING_ZHANGSUEN)
    # return thinned

    return img_data
    return fitContour
    return crop
    return binary
    return gray


if __name__ == '__main__':
    # img_path_test = "./data/111.jpg"
    img_path_test = "./data/002.jpg"
    img_data = cv2.imread(img_path_test)
    img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
    img_data = img_data[296:566, :]
    e1 = cv2.getTickCount()
    # your code execution
    result = get_calc_scale(img_data)
    e2 = cv2.getTickCount()
    time =( (e2 - e1) / cv2.getTickFrequency())*1000
    cv2.namedWindow('123', cv2.WINDOW_NORMAL)
    cv2.imshow("123", result)
    print("time:", time)
    # 加入True判断，使得窗口可持续运行
    while True:
        # if cv2.waitKey(30) & 0xFF == ord('q'):
        if cv2.waitKey(30) & 0xFF == 27:
            break

    cv2.destroyAllWindows()
    # cv2.imwrite("321.jpg",img_data)
    print()
    pass
