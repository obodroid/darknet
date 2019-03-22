from ctypes import *
from multiprocessing import Process
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

    def run(self):
        setproctitle.setproctitle("Dummy Process")
        print("{} Dummy Process Started".format(time_now()))

        receivedDataLength = 0
        while True:
            while not self.queue.empty():
                data = self.queue.get()
                receivedDataLength += len(data)
                print("{} Received: {}".format(time_now(), human_format(len(data))))
                print("{} Cumulative Received: {}".format(time_now(), human_format(receivedDataLength)))

            cv2.waitKey(1)
            sys.stdout.flush()

def human_format(number):
    units = ['', 'K', 'M', 'G', 'T', 'P']
    k = 1000.0
    magnitude = int(floor(log(number, k)))
    return '%.2f%s' % (number / k**magnitude, units[magnitude])

def time_now():
    return datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]