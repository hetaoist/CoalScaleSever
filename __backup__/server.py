#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
# @Time : 2023/8/10 13:46:03
# @Author : Hetao
# @File : [Code] --> scale_server.py
# @Software: PyCharm
# @Function: fastapi service for task.
# -*- coding:utf-8 -*-
import argparse
import configparser

import httpx
import uvicorn
from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse

from pylogger import log

config = configparser.ConfigParser()

config.read("./config.ini", encoding="utf-8")

TIMEOUT_KEEP_ALIVE = 15  # seconds.

app = FastAPI()

pool_limits = httpx.Limits(
        max_connections=1000,
        max_keepalive_connections=1000,
)

client = httpx.Client(limits=pool_limits, timeout=TIMEOUT_KEEP_ALIVE)


@app.post("/run", summary="output")
async def code_translate(req=Body(None)):
    log.logger.debug("start run task:{}".format(req))
    try:
        pass
    except Exception as e:
        res = {"message": e, "result": {}, "status": 0}
        return JSONResponse(res)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=12345)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    uvicorn.run(app="scale_server:app",
                host=args.host,
                port=args.port,
                log_level="debug",
                workers=args.workers,
                reload=False,
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE
                )
