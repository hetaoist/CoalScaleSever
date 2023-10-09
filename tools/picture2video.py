#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
# @Time : 2023/8/20 17:30:21
# @Author : Hetao
# @File : [Code] --> picture2video.py
# @Software: PyCharm
# @Function: 图片转测试视频用于做测试数据
import os

import cv2


def pictures2video(dir_path):
    filelist = os.listdir(dir_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # base_path, video_name = os.path.split(dir_path)
    video = cv2.VideoWriter('{}.avi'.format(dir_path), fourcc, 60, (1280, 1024))
    for name in filelist:
        print(name)
        pic_path = os.path.join(dir_path, name)
        img = cv2.imread(pic_path)
        video.write(img)
    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    _test = "/workspace/Code/hetaoist/scale_sever/data/cut_normal"
    # _test = "/workspace/Code/hetaoist/scale_sever/data/cut_all"
    pictures2video(_test)
    print()
    pass