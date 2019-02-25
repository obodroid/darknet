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
from twisted.internet import task, reactor, threads
from twisted.internet.defer import Deferred, inlineCallbacks
import os
import signal
import sys
import json
from datetime import datetime
from PIL import Image
from io import StringIO
import time
import base64
import logging
from streamVideo import StreamVideo
import benchmark

fileDir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(fileDir, ".."))

# log = logging.getLogger() # 'root' Logger
# console = logging.StreamHandler()
# timeNow = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')
# logFile = logging.FileHandler("/src/benchmark/darknet_bench_{}.log".format(timeNow))
# saveDir = "/src/benchmark/images/"

# format_str = '%(asctime)s\t%(levelname)s -- %(processName)s %(filename)s:%(lineno)s -- %(message)s'
# console.setFormatter(logging.Formatter(format_str))
# logFile.setFormatter(logging.Formatter(format_str))

# log.addHandler(console) # prints to console.
# log.addHandler(logFile) # prints to console.
# log.setLevel(logging.DEBUG) # anything ERROR or above
# log.warn('Import darknet.py!')
# log.critical('Going to load neural network over GPU!')


class Detector(threading.Thread):
    def __init__(self, robotId, videoId, stream, threshold, callback, detectQueue):
        self.threshold = threshold
        self.robotId = robotId
        self.videoId = videoId
        self.stream = stream
        self.video_serial = robotId + "-" + videoId
        self.callback = callback
        self.targetObjects = []
        self.isStop = Value(c_bool, False)
        self.detectQueue = detectQueue

        threading.Thread.__init__(self)
        print ("Detector Initialized {}".format(self.video_serial))

    def run(self):
        head = "data:image/jpeg;base64,"
        if self.stream.startswith(head):
            print("Detector consume image {}".format(self.video_serial))
            imgData = base64.b64decode(self.stream[len(head):])
            imgStr = StringIO.StringIO()
            imgStr.write(imgData)
            imgStr.seek(0)
            imgPIL = Image.open(imgStr)
            frame = cv2.cvtColor(np.asarray(imgPIL), cv2.COLOR_RGB2BGR)
            darknet.putLoad(self, self.keyframe, frame)
            return

        streamVideo = StreamVideo(self.stream, self.video_serial, self.isStop, self.detectQueue)
        streamVideo.start()
        self.videoCaptureReady()
        streamVideo.join()
        self.videoStop()
        print("Detector {} Stopped".format(self.video_serial))

    def stopStream(self):
        self.isStop.value = True
        print("Detector {} stopStream: isStop {} ".format(
            self.video_serial, self.isStop.value))

    def updateTarget(self, targetObjects):
        print("Detector {} updateTarget {}".format(self.video_serial, targetObjects))
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
