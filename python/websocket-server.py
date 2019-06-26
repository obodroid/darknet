# Copyright 2015-2016 Carnegie Mellon University
# Edited 2018 Obodroid Corporation by Lertlove
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# import benchmark
import tracker
import detector
import darknet
import dummyProcess
from ctypes import *
from multiprocessing import Queue, Value
import multiprocessing as mp
import threading
import ssl
from datetime import datetime
import time
import base64
import numpy as np
from PIL import Image
import json
import imagehash
import cv2
import argparse
import urllib
from twisted.python import log
from twisted.internet.ssl import DefaultOpenSSLContextFactory
from twisted.internet.defer import Deferred, inlineCallbacks
from twisted.internet import task, reactor, threads
from autobahn.twisted.websocket import WebSocketServerProtocol, \
    WebSocketServerFactory
import txaio
import pprint
import traceback
import ptvsd

# Allow other computers to attach to ptvsd at this IP address and port, using the secret
# ptvsd.enable_attach("my_secret", address=('0.0.0.0', 3000))
# Pause the program until a remote debugger is attached
# ptvsd.wait_for_attach()

import os
import sys
fileDir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(fileDir, ".."))
txaio.use_twisted()


# For TLS connections
tls_crt = os.path.join(fileDir, 'tls', 'server.crt')
tls_key = os.path.join(fileDir, 'tls', 'server.key')

parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=9000,
                    help='WebSocket Port')
parser.add_argument('--dummy', help="Send dummy text for testing purpose",
                    action="store_true")
args = parser.parse_args()
dummyText = 'x' * int(8.3 * 1000000)


class DarknetServerProtocol(WebSocketServerProtocol):
    def __init__(self):
        super(DarknetServerProtocol, self).__init__()
        self.imageKeyFrame = 0
        self.numWorkers = 1
        self.numGpus = 1
        self.detectors = {}
        self.detectThroughput = Value('i', 0)
        self.detectQueue = Queue(maxsize=10)
        self.detectResultQueue = Queue()
        self.trackers = {}
        self.trackingQueues = {}
        self.trackingResultQueue = Queue()
        self.dummyQueue = Queue(maxsize=10)

    def onConnect(self, request):
        print("Client connecting: {0}".format(request.peer))

    def onOpen(self):
        print("WebSocket connection open.")

    def onMessage(self, payload, isBinary):
        raw = payload.decode('utf8')
        print("server receive msg - {}".format(raw))
        msg = json.loads(raw)

        if msg['type'] == "SETUP":
            if msg['num_workers']:
                self.numWorkers = msg['num_workers']
            if msg['num_gpus']:
                self.numGpus = msg['num_gpus']
            if msg['tracker_gpu_index']:
                self.trackerGpuIndex = msg['tracker_gpu_index']
            if msg['debug']:
                benchmark.enable = True

            t = threading.Thread(target=self.loopTrackResult)
            t.setDaemon(True)
            t.start()

            t = threading.Thread(target=self.loopSendResult)
            t.setDaemon(True)
            t.start()

            self.monitor(0.5)

            if args.dummy:
                print("Create Dummy Process")
                d = dummyProcess.Dummy(self.dummyQueue)
                d.start()

                t = threading.Thread(target=self.loopSendDummy)
                t.setDaemon(True)
                t.start()
            else:
                darknet.initSaveImage()
                darknet.initDarknetWorkers(
                    self.numWorkers, self.numGpus, self.detectQueue, self.detectResultQueue)

            return

        robotId = msg['robotId']
        videoId = msg['videoId']
        video_serial = robotId + "-" + videoId

        if msg['type'] == "START":
            print("START - {}".format(video_serial))
            self.processVideo(msg)
        elif msg['type'] == "IMAGE":
            print("IMAGE - {}".format(video_serial))
            self.processImage(msg)
        elif msg['type'] == "UPDATE":
            print("UPDATE - {}".format(video_serial))
            if video_serial in self.detectors:
                self.detectors[video_serial].updateTarget(msg['options']['targetObjects'])
        elif msg['type'] == "STOP":
            print("STOP - {}".format(video_serial))
            if video_serial in self.detectors:
                self.removeDetector(video_serial)
        elif msg['type'] == "ECHO":
            print("ECHO - {}".format(video_serial))
            if args.dummy:
                # attach message with maximum size limit
                msg['response'] = dummyText
            self.detectCallback(msg)
        elif msg['type'] == "READY":
            print("READY - {}".format(video_serial))
            self.detectCallback(msg)
        else:
            print("Warning: Unknown message type: {}".format(msg['type']))

    def onClose(self, wasClean, code, reason):
        for video_serial in list(self.detectors.keys()):
            self.removeDetector(video_serial)
        darknet.deinitDarknetWorkers()
        print("WebSocket connection closed: {0}".format(reason))

    def processImage(self, msg):
        robotId = msg['robotId']
        videoId = msg['videoId']
        image = msg['stream']
        video_serial = robotId + "-" + videoId

        print("processImage {}".format(video_serial))
        detectorWorker = detector.Detector(
            robotId, videoId, image, None, self.detectCallback, self.detectQueue)

        self.imageKeyFrame += 1
        detectorWorker.keyframe = self.imageKeyFrame
        detectorWorker.start()

    def processVideo(self, msg):
        robotId = msg['robotId']
        videoId = msg['videoId']
        stream = msg['stream']
        threshold = msg['threshold']
        video_serial = robotId + "-" + videoId

        if video_serial in self.detectors:
            print("{} - video is already process".format(video_serial))
            return

        print("processVideo {}".format(video_serial))
        detectorWorker = detector.Detector(
            robotId, videoId, stream, threshold, self.detectCallback, self.detectQueue, self.detectThroughput)
        detectorWorker.setDaemon(True)
        detectorWorker.start()

        isTrackerStop = Value(c_bool, False)
        self.trackingQueues[video_serial] = Queue()
        trackingWorker = tracker.DeepSort(
            video_serial, isTrackerStop, self.trackerGpuIndex, self.trackingQueues[video_serial], self.trackingResultQueue)
        trackingWorker.start()

        self.detectors[video_serial] = detectorWorker
        self.trackers[video_serial] = isTrackerStop

    def detectCallback(self, msg):
        reactor.callFromThread(
            self.sendMessage, json.dumps(msg).encode(), sync=True)

    def removeDetector(self, video_serial):
        self.detectors[video_serial].stopStream()
        del self.detectors[video_serial]
        if video_serial in self.trackers:
            self.trackers[video_serial].value = True

    def doSendResult(self, video_serial, msg):
        if video_serial in self.detectors:
            msg['detectedObjects'] = [detectedObject for detectedObject in msg['detectedObjects']
                                      if detectedObject['objectType'] in self.detectors[video_serial].targetObjects]
        if len(msg['detectedObjects']) > 0:
            self.detectCallback(msg)

    def loopSendDummy(self):
        while True:
            # print("send dummy text at qsize: {}".format(self.dummyQueue.qsize()))
            self.dummyQueue.put(dummyText)
            cv2.waitKey(10)

    def loopTrackResult(self):
        while True:
            while not self.detectResultQueue.empty():
                robotId, videoId, msg, frame, bboxes, confidences = self.detectResultQueue.get()
                video_serial = robotId + "-" + videoId
                if video_serial in self.trackingQueues:
                    self.trackingQueues[video_serial].put(
                        [robotId, videoId, msg, frame, bboxes, confidences])
                else:
                    self.doSendResult(video_serial, msg)
            else:
                cv2.waitKey(1)

    def loopSendResult(self):
        while True:
            while not self.trackingResultQueue.empty():
                robotId, videoId, msg = self.trackingResultQueue.get()
                video_serial = robotId + "-" + videoId
                self.doSendResult(video_serial, msg)
            else:
                cv2.waitKey(1)

    def monitor(self, interval):
        t = threading.Timer(interval, self.monitor, [interval])
        t.setDaemon(True)
        t.start()

        detectDropFrames = {}
        for video_serial in self.detectors:
            detectDropFrames[video_serial] = self.detectors[video_serial].dropFrameCount.value

        msg = {
            'type': 'MONITOR',
            'detectRates': darknet.getDetectRates(),
            'detectDropFrames': detectDropFrames,
            'detectQueueSize': self.detectQueue.qsize(),
            'detectThroughput': self.detectThroughput.value,
        }

        self.detectThroughput.value = 0
        self.detectCallback(msg)


def main(reactor):
    observer = log.startLogging(sys.stdout)
    observer.timeFormat = "%Y-%m-%d %T.%f"
    # txaio.start_logging(level='debug')
    factory = WebSocketServerFactory()
    factory.setProtocolOptions(
        autoPingInterval=1,
        autoPingTimeout=10,
        autoFragmentSize=1000000
    )
    factory.protocol = DarknetServerProtocol
    # ctx_factory = DefaultOpenSSLContextFactory(tls_key, tls_crt)
    # reactor.listenSSL(args.port, factory, ctx_factory)
    reactor.listenTCP(args.port, factory)
    reactor.run()
    return Deferred()


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    q = Queue()
    task.react(main)
