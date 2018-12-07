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
	def __init__(self, path, queueSize=10):
		# initialize the file video stream along with the boolean
		# used to indicate if the thread should be stopped or not
		self.stream = cv2.VideoCapture(path)
		self.fps = self.stream.get(cv2.CAP_PROP_FPS)
		print("run VideoCapture isOpen - {}, fps - {}".format(self.stream.isOpened(),self.fps))    
		self.stopped = False
		self.dropCount = 0
		self.keyframe = 0
		# initialize the queue used to store frames read from
		# the video file
		self.Q = Queue.Queue(maxsize=queueSize)
	def start(self):
        # start a thread to read frames from the file video stream
		t = threading.Thread(target=self.update, args=())
		t.daemon = True
		t.start()
		return self
	
	def update(self):
		# keep looping infinitely
		while True:
			# if the thread indicator variable is set, stop the
			# thread
			if self.stopped:
				return

			self.keyframe += 1
			# otherwise, ensure the queue has room in it
			if self.Q.full():
				self.dropCount += 1
				dropFrame = self.Q.get()
				print("drop frame {} / {}".format(self.keyframe, self.dropCount))

			# read the next frame from the file
			(grabbed, frame) = self.stream.read()

			# if the `grabbed` boolean is `False`, then we have
			# reached the end of the video file
			if not grabbed:
				self.stop()
				return
			
			# add the frame to the queue
			self.Q.put((self.keyframe,frame))
	
	def read(self):
		# return next frame in the queue
		return self.Q.get()
	
	def more(self):
		# return True if there are still frames in the queue
		return self.Q.qsize() > 0
	
	def stop(self):
		# indicate that the thread should be stopped
		self.stopped = True