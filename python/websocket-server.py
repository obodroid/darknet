#!/usr/bin/env python2
#
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
import StringIO
import base64
import time
from datetime import datetime
import ssl
from multiprocessing import Manager
import threading

# For TLS connections
tls_crt = os.path.join(fileDir, 'tls', 'server.crt')
tls_key = os.path.join(fileDir, 'tls', 'server.key')

parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=9000,
                    help='WebSocket Port')
args = parser.parse_args()

import darknet
import detector

class DarknetServerProtocol(WebSocketServerProtocol):
    def __init__(self):
        super(DarknetServerProtocol, self).__init__()
        self.detectors = {}

    def onConnect(self, request):
        print("Client connecting: {0}".format(request.peer))

    def onOpen(self):
        print("WebSocket connection open.")

    def onMessage(self, payload, isBinary):
        raw = payload.decode('utf8')
        print("server raw msg - {}".format(raw))
        msg = json.loads(raw)

        if msg['type'] == "SETUP":
            if msg['debug']:
                darknet.benchmark.enable = True
            return

        robotId = msg['robotId']
        videoId = msg['videoId']
        video_serial = robotId + "-" + videoId

        if msg['type'] == "START":
            print("START - {}".format(video_serial))
            self.processVideo(msg)
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
            self.sendMessage(json.dumps(msg))
        elif msg['type'] == "READY":
            print("READY - {}".format(video_serial))
            self.sendMessage(json.dumps(msg))
        else:
            print("Warning: Unknown message type: {}".format(msg['type']))

    def onClose(self, wasClean, code, reason):
        for video_serial in self.detectors.keys():
            self.removeDetector(video_serial)
        print("WebSocket connection closed: {0}".format(reason))

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
            robotId, videoId, stream, threshold, self.detectCallback)
        detectorWorker.setDaemon(True)
        detectorWorker.start()
        self.detectors[video_serial] = detectorWorker

    def detectCallback(self, msg):
        self.sendMessage(json.dumps(msg), sync=True)
    
    def removeDetector(self,video_serial):
        self.detectors[video_serial].stopStream()
        self.detectors[video_serial].join()
        del self.detectors[video_serial]


def main(reactor):
    observer = log.startLogging(sys.stdout)
    observer.timeFormat = "%Y-%m-%d %T.%f"
    factory = WebSocketServerFactory()
    factory.setProtocolOptions(autoPingInterval=1)
    factory.protocol = DarknetServerProtocol
    # ctx_factory = DefaultOpenSSLContextFactory(tls_key, tls_crt)
    # reactor.listenSSL(args.port, factory, ctx_factory)
    reactor.listenTCP(args.port, factory)
    reactor.run()
    return Deferred()

if __name__ == '__main__':
    task.react(main)
