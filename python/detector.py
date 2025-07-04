from ctypes import *
import math
import random
import argparse
import cv2
import numpy as np
import queue
import threading
from multiprocessing import Value
from random import randint
from threading import Timer
import os
import signal
import sys
import json
from datetime import datetime
import time
import base64
import logging
from streamVideo import StreamVideo
# import benchmark

fileDir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(fileDir, ".."))


class Detector(threading.Thread):
    def __init__(self, robotId, videoId, stream, callback, detectQueue, detectThroughput):
        self.robotId = robotId
        self.videoId = videoId
        self.stream = stream
        self.video_serial = robotId + "-" + videoId
        self.keyframe = 1
        self.callback = callback
        self.targetObjects = []
        self.isStop = Value(c_bool, False)
        self.dropFrameCount = Value('i', 0)
        self.detectQueue = detectQueue
        self.detectThroughput = detectThroughput

        threading.Thread.__init__(self)
        print ("Detector Initialized {}".format(self.video_serial))

    def run(self):
        head = "data:image/jpeg;base64,"
        if self.stream.startswith(head):
            print("Detector consume image {}".format(self.video_serial))
            imgData = base64.b64decode(self.stream[len(head):])
            imgArr = np.fromstring(imgData, np.uint8)
            frame = cv2.imdecode(imgArr, cv2.IMREAD_COLOR)
            self.detectQueue.put([self.video_serial, self.keyframe, frame, datetime.now()])
            print("Detector {} detectQueue qsize: {}".format(self.video_serial, self.detectQueue.qsize()))
            return

        streamVideo = StreamVideo(
            self.stream, self.video_serial, self.isStop, not self.stream.endswith("mp4"), self.dropFrameCount, self.detectQueue, self.detectThroughput)
        streamVideo.start()
        self.videoCaptureReady()
        streamVideo.join()
        self.videoStop()
        print("Detector {} Stopped".format(self.video_serial))

    def stopStream(self):
        print("Detector {} stopStream".format(self.video_serial))
        self.isStop.value = True

    def updateTarget(self, targetObjects):
        print("Detector {} updateTarget {}".format(
            self.video_serial, targetObjects))
        self.targetObjects = targetObjects

    def videoCaptureReady(self):
        msg = {
            "type": "READY",
            "robotId": self.robotId,
            "videoId": self.videoId,
        }
        self.callback(msg)

    def videoStop(self):
        msg = {
            "type": "STOP",
            "robotId": self.robotId,
            "videoId": self.videoId,
        }
        self.callback(msg)

    def sendLogMessage(self, keyframe, step):
        msg = {
            "type": "LOG",
            "robotId": self.robotId,
            "videoId": self.videoId,
            "keyframe": keyframe,
            "step": step,
            "time": datetime.now().isoformat()
        }
        if benchmark.enable:
            self.callback(msg)
