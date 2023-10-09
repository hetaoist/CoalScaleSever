#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
# @Time : 2023/8/19 09:52:05
# @Author : Taylor
# @File : [Code] --> run.py
# @Software: PyCharm
# @Function: 程序启动入口，用uvicorn脚本启动该文件。
import argparse
import configparser
import time

import gol
import uvicorn

from deal import video_deal
from pylogger import log

config = configparser.ConfigParser()
config.read("./config.ini", encoding="utf-8")


def start_run(debug):
    while True:
        # 初始化全局变量
        gol.init()
        # 没检测到视频则程序持续检测不退出
        video_deal(debug)
        log.logger.error("No cameras found will be detected every 5 seconds...")
        time.sleep(5)


# 启动服务
start_run(debug=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=12345)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    uvicorn.run(app="run:app",
                host=args.host,
                port=args.port,
                log_level="debug",
                workers=args.workers,
                reload=False,
                timeout_keep_alive=15
                )
