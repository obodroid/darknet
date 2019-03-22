from ctypes import *
from imutils.video import FPS
import imutils
import argparse
import cv2
import numpy as np
import threading
from multiprocessing import Process, Queue
import os
import signal
import sys
import json
from datetime import datetime
import time
import base64


class StreamVideo(Process):
    def __init__(self, path, video_serial, isStop, detectQueue):
        Process.__init__(self)
        self.daemon = True
        self.path = path
        self.video_serial = video_serial
        self.max_fps = 4.0
        self.fps = None
        self.isStop = isStop
        self.isDisplay = False
        self.dropCount = 0
        self.keyframe = 0
        self.previous_frame_time = datetime.now()
        self.detectQueue = detectQueue

    def putLoad(self, videoSerial, keyframe, frame, time):
        print("darknet detectQueue qsize: {}".format(self.detectQueue.qsize()))
        if self.detectQueue.full():
            dropVideoSerial, _, _, _ = self.detectQueue.get()
            print("darknet drop frame {}, keyframe {}".format(dropVideoSerial, keyframe))
        self.detectQueue.put([videoSerial, keyframe, frame, time])

    def run(self):
        self.stream = cv2.VideoCapture(self.path)

        if self.stream.isOpened():
            self.fps = self.stream.get(cv2.CAP_PROP_FPS)
            print("StreamVideo {} run VideoCapture at path {}, fps {}".format(self.video_serial, self.path, self.fps))
        else:
            print("StreamVideo {} error VideoCapture at path {}".format(self.video_serial, self.path))
            self.stop()

        # fps = FPS().start()
        while self.isStop.value is False:
            self.keyframe += 1

            # print("StreamVideo {} start read stream".format(self.video_serial))
            (grabbed, frame) = self.stream.read()
            # print("StreamVideo {} finish read stream".format(self.video_serial))

            current_frame_time = datetime.now()
            diff_from_previous_frame = (current_frame_time - self.previous_frame_time).total_seconds()
            if diff_from_previous_frame < (1.0 / self.max_fps):
                self.dropCount += 1
                # print("StreamVideo {} drop high fps frame at keyframe {} / drop count {} / time diff {}".format(
                #     self.video_serial, self.keyframe, self.dropCount, diff_from_previous_frame))
                continue

            if not grabbed:
                self.stop()
                return

            # if self.isDisplay:
            #     displayScreen = "video : {}".format(self.video_serial)
            #     cv2.imshow(displayScreen, frame)
            frameRs=cv2.resize(frame, (1280,720))
            cv2.imshow("showFrame", frameRs)
            key=cv2.waitKey(10)

            self.putLoad(self.video_serial, self.keyframe, frame, current_frame_time)
            self.previous_frame_time = current_frame_time
            sys.stdout.flush()

        # fps.stop()
        self.stream.release()
        print("StreamVideo {} exit".format(self.video_serial))

    def stop(self):
        print("StreamVideo {} stop".format(self.video_serial))
        self.isStop.value = True
