from imutils.video import FPS
from random import randint
import os, signal
import sys
from datetime import datetime
import time
import logging

log = logging.getLogger() # 'root' Logger
console = logging.StreamHandler()
timeNow = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')
logFile = logging.FileHandler("/src/benchmark/darknet_bench_{}.log".format(timeNow))
saveDir = "/src/benchmark/images/"

format_str = '%(asctime)s\t%(levelname)s -- %(processName)s %(filename)s:%(lineno)s -- %(message)s'
console.setFormatter(logging.Formatter(format_str))
logFile.setFormatter(logging.Formatter(format_str))

log.addHandler(console) # prints to console.
log.addHandler(logFile) # prints to console.
log.setLevel(logging.DEBUG) # anything ERROR or above
# log.warn('Import darknet.py!')
# log.critical('Going to load neural network over GPU!')

mode = ''
benchmarks = {}
imageCount = 0

def startBenchmark(period,tag):
    if tag not in benchmarks and mode == 'benchmark' :
        print("startBenchmark {}".format(tag))
        fps = FPS().start()
        benchmarks[tag] = fps
        t = Timer(period, endBenchmark,[fps,tag])
        t.start() # after 30 seconds, "hello, world" will be printed

def updateBenchmark(tag):
    # print("updateBenchmark {}".format(tag))
    if tag in benchmarks:
        benchmarks[tag].update()

def endBenchmark(fps,tag):
    print("endBenchmark {}".format(tag))
    fps.stop()
    log.info("{} rate: {:.2f}".format(tag,fps.fps()))
    if tag in benchmarks:
        del benchmarks[tag]