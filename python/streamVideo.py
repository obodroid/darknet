from ctypes import *
from imutils.video import FPS
import imutils
import argparse
import cv2
import numpy as np
import threading
from multiprocessing import Process, Queue
import setproctitle
import os
import signal
import sys
import json
from datetime import datetime
import time
import base64


class StreamVideo(Process):
    def __init__(self, path, video_serial, isStop, isLive, dropFrameCount, detectQueue, detectThroughput):
        Process.__init__(self)
        self.daemon = True
        self.path = path
        self.video_serial = video_serial
        self.max_fps = 4.0
        self.fps = None
        self.isStop = isStop
        self.isDisplay = False
        self.isLive = isLive
        self.dropCount = 0
        self.keyframe = 0
        self.previous_frame_time = datetime.now()
        self.dropFrameCount = dropFrameCount
        self.detectQueue = detectQueue
        self.detectThroughput = detectThroughput

    def putLoad(self, videoSerial, keyframe, frame, time):
        print("StreamVideo {} detectQueue qsize: {}".format(
            videoSerial, self.detectQueue.qsize()))
        if self.detectQueue.full():
            print("StreamVideo {} drop frame, keyframe {}".format(
                videoSerial, keyframe))
            self.dropFrameCount.value += 1
            return
        self.detectQueue.put([videoSerial, keyframe, frame, time])
        self.detectThroughput.value += frame.nbytes

    def run(self):
        setproctitle.setproctitle("Stream Video {}".format(self.video_serial))

        former_capture_options = os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"]
        if self.path.endswith("sdp") and former_capture_options.find("hevc_cuvid") > -1:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = former_capture_options.replace("hevc_cuvid","h264_cuvid")

        print("StreamVideo {} run VideoCapture at path {} with {}".format(
            self.video_serial, self.path, os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"]))
        self.stream = cv2.VideoCapture(self.path)

        if self.stream.isOpened():
            self.fps = self.stream.get(cv2.CAP_PROP_FPS)
            print("StreamVideo {} open VideoCapture at path {}, fps {}".format(
                self.video_serial, self.path, self.fps))
        else:
            print("StreamVideo {} error VideoCapture at path {} => try udp rtsp transport".format(
                self.video_serial, self.path))

            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"].replace("tcp","udp")
            self.stream = cv2.VideoCapture(self.path)
            if self.stream.isOpened():
                self.fps = self.stream.get(cv2.CAP_PROP_FPS)
                print("StreamVideo {} run VideoCapture at path {}, fps {} with {}".format(
                    self.video_serial, self.path, self.fps, os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"]))
            else:
                print("StreamVideo {} error VideoCapture at path {} => stop capturing".format(
                    self.video_serial, self.path))
                self.stop()

        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = former_capture_options

        # fps = FPS().start()
        while self.isStop.value is False:
            self.keyframe += 1

            # print("StreamVideo {} start read stream".format(self.video_serial))
            (grabbed, frame) = self.stream.read()
            # print("StreamVideo {} finish read stream".format(self.video_serial))

            current_frame_time = datetime.now()
            diff_from_previous_frame = (
                current_frame_time - self.previous_frame_time).total_seconds()
            
            if self.isLive and diff_from_previous_frame < (1.0 / self.max_fps):
                self.dropCount += 1
                # print("StreamVideo {} drop high fps frame at keyframe {} / drop count {} / time diff {}".format(
                #     self.video_serial, self.keyframe, self.dropCount, diff_from_previous_frame))
                continue

            if not grabbed:
                print("StreamVideo {} grab failed".format(self.video_serial))
                self.stop()
                continue

            if frame.shape[1] > 1080:
                # resize image if width is larger than 1080
                print("StreamVideo {} original frame size {}".format(
                    self.video_serial, frame.shape))
                
                scale_percent = 1080 / frame.shape[1]
                width = int(frame.shape[1] * scale_percent)
                height = int(frame.shape[0] * scale_percent)
                dim = (width, height)
                frame = cv2.resize(frame, dim, interpolation = cv2.INTER_NEAREST)

            if self.isDisplay:
                print("StreamVideo {} show frame {}".format(self.video_serial, self.keyframe))
                title = "video : {}".format(self.video_serial)
                cv2.imshow(title, frame)
                cv2.waitKey(1)

            self.putLoad(self.video_serial, self.keyframe,
                         frame, current_frame_time)
            self.previous_frame_time = current_frame_time
            sys.stdout.flush()

            if not self.isLive:
                cv2.waitKey(int(1000 / self.fps))

        # fps.stop()
        self.stream.release()
        print("StreamVideo {} exit".format(self.video_serial))

    def stop(self):
        print("StreamVideo {} stop".format(self.video_serial))
        self.isStop.value = True
