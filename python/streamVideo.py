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
        print("init stream video at path - {}".format(path))

        if self.stream.isOpened():
            self.fps = self.stream.get(cv2.CAP_PROP_FPS)
            print("run VideoCapture fps - {}, path - {}".format(self.fps, path))
            self.stopped = False
            self.dropCount = 0
            self.keyframe = 0
            # initialize the queue used to store frames read from the video file
            self.captureQueue = Queue.Queue(maxsize=queueSize)
            self.fetchWorker = threading.Thread(target=self.update, args=())
            self.fetchWorker.setDaemon(True)
            self.fetchWorker.start()
        else:
            print("error VideoCapture at path - {}".format(path))
            self.stopped = True

    def update(self):
        # keep looping infinitely
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            self.keyframe += 1
            # otherwise, ensure the queue has room in it

            print("captureQueue {} qsize: {}".format(self.video_serial, self.captureQueue.qsize()))
            if self.captureQueue.full():
                self.dropCount += 1
                self.captureQueue.get()
                print(
                    "drop frame {}, keyframe {} / drop count {}".format(self.video_serial, self.keyframe, self.dropCount))

            # read the next frame from the file
            (grabbed, frame) = self.stream.read()

            # if the `grabbed` boolean is `False`, then we have
            # reached the end of the video file
            if not grabbed:
                self.stop()
                return

            # add the frame to the queue
            self.captureQueue.put((self.keyframe, frame))

    def read(self):
        # return next frame in the queue
        return self.captureQueue.get()

    def more(self):
        # return True if there are still frames in the queue
        return self.captureQueue.qsize() > 0

    def stop(self):
        # indicate that the thread should be stopped
        print("stop VideoCapture - {}".format(self.video_serial))
        self.stopped = True
