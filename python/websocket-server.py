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

import ptvsd

# Allow other computers to attach to ptvsd at this IP address and port, using the secret
# ptvsd.enable_attach("my_secret", address=('0.0.0.0', 3000))
# Pause the program until a remote debugger is attached
# ptvsd.wait_for_attach()

import os
import sys
fileDir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(fileDir, ".."))
import traceback
import pprint
import txaio
txaio.use_twisted()

from autobahn.twisted.websocket import WebSocketServerProtocol, \
    WebSocketServerFactory
from twisted.internet import task, reactor, threads
from twisted.internet.defer import Deferred, inlineCallbacks
from twisted.internet.ssl import DefaultOpenSSLContextFactory
from twisted.python import log

import urllib
import argparse
import cv2
import imagehash
import json
from PIL import Image
import numpy as np
import base64
import time
from datetime import datetime
import ssl
import threading
import multiprocessing as mp
from multiprocessing import Queue

# For TLS connections
tls_crt = os.path.join(fileDir, 'tls', 'server.crt')
tls_key = os.path.join(fileDir, 'tls', 'server.key')

parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=9000,
                    help='WebSocket Port')
args = parser.parse_args()

import darknet
import detector
import benchmark

dummyText = ''
for i in range(0, 8300000):
    dummyText += str(1)

darknetIsInit = False

class DarknetServerProtocol(WebSocketServerProtocol):
    def __init__(self):
        super(DarknetServerProtocol, self).__init__()
        self.imageKeyFrame = 0
        self.numWorkers = 1
        self.numGpus = 1
        self.detectors = {}
        self.detectQueue = Queue(maxsize=10)

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
            if msg['debug']:
                benchmark.enable = True

            self.monitor(0.5)
            
            global darknetIsInit
            print("server darknetIsInit - {}".format(darknetIsInit))
            if not darknetIsInit:
                darknetIsInit = True
                darknet.initSaveImage()
                darknet.initDarknetWorkers(self.numWorkers, self.numGpus, self.detectQueue)
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
                self.detectors[video_serial].updateTarget(msg['targetObjects'])
        elif msg['type'] == "STOP":
            print("STOP - {}".format(video_serial))
            if video_serial in self.detectors:
                self.removeDetector(video_serial)
        elif msg['type'] == "ECHO":
            print("ECHO - {}".format(video_serial))
            if False:
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
        print("WebSocket connection closed: {0}".format(reason))

    def processImage(self, msg):
        robotId = msg['robotId']
        videoId = msg['videoId']
        image = msg['stream']
        video_serial = robotId + "-" + videoId

        print("processImage {}".format(video_serial))
        detectorWorker = detector.Detector(
            robotId, videoId, image, None, self.detectCallback)
        
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
            robotId, videoId, stream, threshold, self.detectCallback, self.detectQueue)
        detectorWorker.setDaemon(True)
        detectorWorker.start()
        self.detectors[video_serial] = detectorWorker

    def detectCallback(self, msg):
        reactor.callFromThread(self.sendMessage, json.dumps(msg).encode(), sync=True)
    
    def removeDetector(self,video_serial):
        self.detectors[video_serial].stopStream()
        del self.detectors[video_serial]

    def monitor(self, interval):
        t = threading.Timer(interval, self.monitor, [interval])
        t.setDaemon(True)
        t.start()
        msg = {
            'type': 'MONITOR',
            'detectRates': darknet.getDetectRates(),
            'detectQueueSize': self.detectQueue.qsize(),
        }
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
