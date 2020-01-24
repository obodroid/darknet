from autobahn.twisted.websocket import WebSocketClientProtocol
import json
import argparse
import threading
import time
from threading import Timer

parser = argparse.ArgumentParser()
parser.add_argument('--setup', type=bool, default=True,
                    help='If True, initialize darknet worker. Only stream, if False')
parser.add_argument('--numWorkers', type=int, default=1,
                    help='num workers')
parser.add_argument('--robotId', type=str, default="1",
                    help='robotId')
parser.add_argument('--videoId', type=str, default="1",
                    help='videoId')
parser.add_argument('--stream', type=str, default="rtsp://admin:123456@192.168.1.13/ch01.264?ptype=udp",
                    help='stream')

args = parser.parse_args()

class DarknetClientProtocol(WebSocketClientProtocol):

    def onOpen(self):
        if args.setup:
            self.setup()
        self.startStream(args.stream, args.robotId, args.videoId)
        nextStart = Timer(10.0, self.startStream, (args.stream,"1","1"))
        nextStart.start()

    def setup(self):
        print("Setup darknet config")
        numWorkers = args.numWorkers
        numGpus = 1 # aarch64 has only 1 gpu
        msg = {
            "type":"SETUP",
            "num_workers":numWorkers,
            "num_gpus":numGpus,
            "debug":False
        }
        jsonMsg = json.dumps(msg)
        self.sendMessage(jsonMsg.encode('utf8'))

    def startStream(self,stream,robotId,videoId):
        msg = {
                "type":"START",
                "stream":stream,
                "robotId":robotId,
                "videoId":videoId,
                "threshold":"0.5"
            }
        jsonMsg = json.dumps(msg)
        self.sendMessage(jsonMsg.encode('utf8'))

    def onMessage(self, payload, isBinary):
        if isBinary:
            print("Binary message received: {0} bytes".format(len(payload)))
        else:
            print("Text message received: {0}".format(payload.decode('utf8')))

if __name__ == '__main__':

    import sys

    from twisted.python import log
    from twisted.internet import reactor
    log.startLogging(sys.stdout)

    from autobahn.twisted.websocket import WebSocketClientFactory
    factory = WebSocketClientFactory()
    factory.protocol = DarknetClientProtocol

    reactor.connectTCP("127.0.0.1", 9000, factory)
    reactor.run()