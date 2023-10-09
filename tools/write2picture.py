#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
# @Time : 2023/8/19 09:52:05
# @Author : Hetao
# @File : [Code] --> run.py
# @Software: PyCharm
# @Function: 读取
import cv2
# import configparser

# from roi import get_line_roi
# import numpy as np

# config = configparser.ConfigParser()
# config.read("./config.ini", encoding="utf-8")

cap = cv2.VideoCapture('./001.avi')


if (cap.isOpened()== False):
  print("Error opening video stream or file")
count = 1
while(cap.isOpened()):
  ret, frame = cap.read()
  if ret == True:
    img_path = "/workspace/Code/hetaoist/scale_sever/data/001/{:0>6d}.jpg".format(count)
    cv2.imwrite(img_path, frame)
    print(img_path)
    if cv2.waitKey(30) & 0xFF == ord('q'):
      break
    count += 1
  else:
    break

cap.release()

cv2.destroyAllWindows()
