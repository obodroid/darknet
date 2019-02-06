from ctypes import *
from imutils.video import FPS
import imutils
import argparse
import cv2
import numpy as np
import threading
import Queue
from multiprocessing import Process
import os
import signal
import sys
import json
from datetime import datetime
import time
import base64


class StreamVideo:
    def __init__(self, path, video_serial, queueSize=10):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.stream = cv2.VideoCapture(path)
        self.video_serial = video_serial
        self.max_fps = 4.0

        if self.stream.isOpened():
            self.fps = self.stream.get(cv2.CAP_PROP_FPS)
            print(
                "StreamVideo run VideoCapture at path - {}, fps - {}".format(path, self.fps))
            self.stopped = False
            self.dropCount = 0
            self.keyframe = 0
            self.previous_frame_time = datetime.now()
            # initialize the queue used to store frames read from the video file
            self.captureQueue = Queue.Queue(maxsize=queueSize)
            self.fetchWorker = threading.Thread(target=self.update, args=())
            self.fetchWorker.setDaemon(True)
            self.fetchWorker.start()
        else:
            print("StreamVideo error VideoCapture at path - {}".format(path))
            self.stopped = True

    def update(self):
        # keep looping infinitely
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            self.keyframe += 1

            print("StreamVideo captureQueue {}, keyframe {}, qsize: {}".format(
                self.video_serial, self.keyframe, self.captureQueue.qsize()))
            if self.captureQueue.full():
                self.dropCount += 1
                self.captureQueue.get()
                print("StreamVideo drop queue full frame {}, keyframe {} / drop count {}".format(
                    self.video_serial, self.keyframe, self.dropCount))

            # read the next frame from the file
            # print("StreamVideo start read stream {}".format(self.video_serial))
            (grabbed, frame) = self.stream.read()
            # print("StreamVideo finish read stream {}".format(self.video_serial))

            current_frame_time = datetime.now()
            diff_from_previous_frame = (current_frame_time - self.previous_frame_time).total_seconds()
            if diff_from_previous_frame < (1.0 / self.max_fps):
                self.dropCount += 1
                print("StreamVideo drop high fps frame {}, keyframe {} / drop count {} / time diff {}".format(
                    self.video_serial, self.keyframe, self.dropCount, diff_from_previous_frame))
                continue

            # if the `grabbed` boolean is `False`, then we have
            # reached the end of the video file
            if not grabbed:
                self.stop()
                return

            # add the frame to the queue
            self.captureQueue.put((self.keyframe, frame, current_frame_time))
            self.previous_frame_time = current_frame_time

    def read(self):
        # return next frame in the queue
        try:
            return self.captureQueue.get(False)
        except Queue.Empty:
            return None, None, None

    def stop(self):
        # indicate that the thread should be stopped
        print("StreamVideo stop VideoCapture - {}".format(self.video_serial))
        self.stopped = True
