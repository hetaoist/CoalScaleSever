#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
# @Time : 2023/8/21 13:37:46
# @Author : Taylor
# @File : [Code] --> gol.py
# @Software: PyCharm
# @Function: 全局变量

from pylogger import log


def init():
    global _global_dict
    _global_dict = {}
    # 定义工作状态，True表示面积计算服务正常工作。
    _global_dict["normal"] = False
    _global_dict["end_points_list"] = []
    _global_dict["scale_list"] = []
    _global_dict["position_list"] = []


def set_value(key, value):
    _global_dict[key] = value


def get_value(key):
    try:
        return _global_dict[key]
    except KeyError as e:
        log.logger.error(e)
        raise
