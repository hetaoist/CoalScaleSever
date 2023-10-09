#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
# @Time : 2023/8/21 13:38:34
# @Author : Taylor
# @File : [Code] --> get_logger.py
# @Software: PyCharm
# @Function: 日志配置
import configparser
import logging
from logging import handlers


class PyLogger(object):
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }

    _instance = None

    def __new__(cls, *args, **kwargs):
        """new实现单例模式"""
        if cls._instance is None:
            cls._instance = object.__new__(cls)
        return cls._instance

    def __init__(self, level='info', when='D', back_count=7):
        conf = configparser.ConfigParser()
        conf.read("./config.ini", encoding="utf-8")

        log_path = conf.get('logger', 'log_path')
        log_server_name = conf.get('logger', 'log_server_name')
        fmt = conf.get('logger', 'log_formatter', raw=True)

        self.logger = logging.getLogger(name=log_server_name)
        format_str = logging.Formatter(fmt)
        self.logger.setLevel(self.level_relations.get(level))
        if not self.logger.handlers:
            sh = logging.StreamHandler()
            sh.setFormatter(format_str)
            th = handlers.TimedRotatingFileHandler(filename=log_path,
                                                   when=when,
                                                   backupCount=back_count,
                                                   encoding='utf-8'
                                                   )

            th.setFormatter(format_str)

            self.logger.addHandler(sh)
            self.logger.addHandler(th)


log = PyLogger(level='debug')

