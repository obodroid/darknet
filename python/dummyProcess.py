from ctypes import *
from multiprocessing import Process
import threading
import setproctitle
import numpy as np
import cv2
import os
import sys
from math import log, floor
from datetime import datetime


class Dummy(Process):
    def __init__(self, queue):
        Process.__init__(self)
        self.daemon = True
        self.queue = queue
        self.receivedDataLength = 0

    def run(self):
        setproctitle.setproctitle("Dummy Process")
        print("{} Dummy Process Started".format(time_now()))

        self.monitorDataRate()

        while True:
            while not self.queue.empty():
                data = self.queue.get()
                self.receivedDataLength += len(data)
                # print("{} Received: {}".format(time_now(), human_format(len(data))))
            else:
                cv2.waitKey(1)
            sys.stdout.flush()
    
    def monitorDataRate(self):
        t = threading.Timer(1.0, self.monitorDataRate)
        t.setDaemon(True)
        t.start()
        print("{} Data Queue Size: {}".format(time_now(), self.queue.qsize()))
        print("{} Data Rate: {}Bytes/s".format(time_now(), human_format(self.receivedDataLength)))
        self.receivedDataLength = 0

def human_format(number):
    if number <= 0:
        return '0'
    units = ['', 'K', 'M', 'G', 'T', 'P']
    k = 1000.0
    magnitude = int(floor(log(number, k)))
    return '%.2f%s' % (number / k**magnitude, units[magnitude])

def time_now():
    return datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]