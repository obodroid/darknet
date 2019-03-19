from ctypes import *
from imutils.video import FPS
import math
import random
import argparse
import cv2
import numpy as np
import threading
from multiprocessing import Process, Value
from random import randint
from threading import Timer
from twisted.internet import task, reactor, threads
from twisted.internet.defer import Deferred, inlineCallbacks
import setproctitle
import os
import signal
import sys
import json
from datetime import datetime
import time
import base64
import logging
# import benchmark

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet

fileDir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(fileDir, ".."))

# log = logging.getLogger() # 'root' Logger
# console = logging.StreamHandler()
# timeNow = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')
# logFile = logging.FileHandler("/src/benchmark/darknet_bench_{}.log".format(timeNow))
# saveDir = "/src/benchmark/images/"

# format_str = '%(asctime)s\t%(levelname)s -- %(processName)s %(filename)s:%(lineno)s -- %(message)s'
# console.setFormatter(logging.Formatter(format_str))
# logFile.setFormatter(logging.Formatter(format_str))

# log.addHandler(console) # prints to console.
# log.addHandler(logFile) # prints to console.
# log.setLevel(logging.DEBUG) # anything ERROR or above
# log.warn('Import darknet.py!')
# log.critical('Going to load neural network over GPU!')


def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1


def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr


class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]


class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]


class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


class IplROI(Structure):
    pass


class IplTileInfo(Structure):
    pass


class IplImage(Structure):
    pass


IplImage._fields_ = [
    ('nSize', c_int),
    ('ID', c_int),
    ('nChannels', c_int),
    ('alphaChannel', c_int),
    ('depth', c_int),
    ('colorModel', c_char * 4),
    ('channelSeq', c_char * 4),
    ('dataOrder', c_int),
    ('origin', c_int),
    ('align', c_int),
    ('width', c_int),
    ('height', c_int),
    ('roi', POINTER(IplROI)),
    ('maskROI', POINTER(IplImage)),
    ('imageId', c_void_p),
    ('tileInfo', POINTER(IplTileInfo)),
    ('imageSize', c_int),
    ('imageData', c_char_p),
    ('widthStep', c_int),
    ('BorderMode', c_int * 4),
    ('BorderConst', c_int * 4),
    ('imageDataOrigin', c_char_p)]


class iplimage_t(Structure):
    _fields_ = [('ob_refcnt', c_ssize_t),
                ('ob_type',  py_object),
                ('a', POINTER(IplImage)),
                ('data', py_object),
                ('offset', c_size_t)]


lib = CDLL("/src/darknet/libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int,
                              c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

configPath = b"/src/darknet/cfg/yolov3.cfg"
weightPath = b"/src/data/yolo/yolov3.weights"
metaPath = b"/src/darknet/cfg/coco.data"
imgEncPath = b"/src/data/deep_sort/mars-small128.pb"
thresh = .6
hier_thresh = .5
nms = .45
bufferSize = 3
maxTimeout = 10  # secs

isNeedSaveImage = True
fullImageDir = "/tmp/.robot-app/full_images"


class Darknet(Process):
    def __init__(self, index, numGpus, detectQueue, resultQueue):
        Process.__init__(self)
        self.daemon = True
        self.net = None
        self.meta = None
        self.encoder = None
        self.tracker = None
        self.index = index
        self.numGpus = numGpus
        self.detectCount = 0
        self.detectRate = Value('i', 0)
        self.detectQueue = detectQueue
        self.resultQueue = resultQueue
        self.isStop = Value(c_bool, False)

    def run(self):
        setproctitle.setproctitle("Darknet {}".format(self.index))
        gpuIndex = (self.index % self.numGpus) + \
            ((int(os.environ['CONTAINER_INDEX']) - 1) * self.numGpus) + 1
        set_gpu(gpuIndex)
        print("Load darknet worker = {} with gpuIndex = {}".format(
            self.index, gpuIndex))
        self.net = load_net(configPath, weightPath, 0)
        self.meta = load_meta(metaPath)
        for i in range(self.meta.classes):
            self.meta.names[i] = self.meta.names[i].decode().replace(
                ' ', '_').encode()
        print("darknet {} initialized".format(self.index))

        # Definition of the parameters
        max_cosine_distance = 0.3
        nn_budget = None

        # deep_sort
        self.encoder = gdet.create_box_encoder(imgEncPath, batch_size=1)
        metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric)

        self.monitorDetectRate()

        while not self.isStop.value:
            try:
                video_serial, keyframe, frame, time = self.detectQueue.get(
                    timeout=0.1)
                self.nnDetect(video_serial, keyframe, frame, time)
            except Exception:
                pass
            sys.stdout.flush()

    def monitorDetectRate(self):
        t = threading.Timer(1.0, self.monitorDetectRate)
        t.setDaemon(True)
        t.start()
        self.detectRate.value = self.detectCount
        self.detectCount = 0

    # # original import darknet with some test function
    #
    # def classify(net, meta, im):
    #     out = predict_image(net, im)
    #     res = []
    #     for i in range(meta.classes):
    #         res.append((meta.names[i], out[i]))
    #     res = sorted(res, key=lambda x: -x[1])
    #     return res

    # def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    #     """if isinstance(image, bytes):
    #         # image is a filename
    #         # i.e. image = b'/darknet/data/dog.jpg'
    #         im = load_image(image, 0, 0)
    #     else:
    #         # image is an nparray
    #         # i.e. image = cv2.imread('/darknet/data/dog.jpg')
    #         im, image = array_to_image(image)
    #         rgbgr_image(im)
    #     """
    #     im, image = array_to_image(image)
    #     rgbgr_image(im)
    #     num = c_int(0)
    #     pnum = pointer(num)
    #     predict_image(net, im)
    #     dets = get_network_boxes(net, im.w, im.h, thresh,
    #                             hier_thresh, None, 0, pnum)
    #     num = pnum[0]
    #     if nms:
    #         do_nms_obj(dets, num, meta.classes, nms)

    #     res = []
    #     for j in range(num):
    #         a = dets[j].prob[0:meta.classes]
    #         if any(a):
    #             ai = np.array(a).nonzero()[0]
    #             for i in ai:
    #                 b = dets[j].bbox
    #                 res.append((meta.names[i], dets[j].prob[i],
    #                             (b.x, b.y, b.w, b.h)))

    #     res = sorted(res, key=lambda x: -x[1])
    #     if isinstance(image, bytes):
    #         free_image(im)
    #     free_detections(dets, num)
    #     return res

    def nnDetect(self, video_serial, keyframe, frame, time):
        self.detectCount += 1
        print("darknet {} nnDetect {}, keyframe {}".format(
            self.index, video_serial, keyframe))
        robotId, videoId = video_serial.split('-')
        # red for palmup --> stop, green for thumbsup --> go
        classes_box_colors = [(0, 0, 255), (0, 255, 0)]
        classes_font_colors = [(255, 255, 0), (0, 255, 255)]

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im, arr = array_to_image(rgb_frame)

        num = c_int(0)
        pnum = pointer(num)
        predict_image(self.net, im)
        dets = get_network_boxes(self.net, im.w, im.h, thresh,
                                 hier_thresh, None, 0, pnum)
        num = pnum[0]
        foundObject = False
        filename = '{}_{}.jpg'.format(video_serial, keyframe)
        filepath = '{}/{}'.format(fullImageDir, filename)

        # detector.sendLogMessage(keyframe, "feed_network")

        if (nms):
            do_nms_obj(dets, num, self.meta.classes, nms)

        bboxes = []
        confidences = []
        objectTypes = []
        dataURLs = []

        for j in range(num):
            for i in range(self.meta.classes):
                objectType = self.meta.names[i].decode()
                if dets[j].prob[i] > 0:
                    b = dets[j].bbox
                    x1 = int(b.x - b.w / 2.)
                    y1 = int(b.y - b.h / 2.)
                    x2 = int(b.x + b.w / 2.)
                    y2 = int(b.y + b.h / 2.)

                    x1 = x1 if x1 >= 0 else 0
                    y1 = y1 if y1 >= 0 else 0
                    x2 = x2 if x2 <= im.w else im.w
                    y2 = y2 if y2 <= im.h else im.h

                    # if need to show rectangular bounding box and text, you can uncomment here
                    # cv2.rectangle(frame, (x1, y1), (x2, y2), classes_box_colors[1], 2)
                    # cv2.putText(frame, self.meta.names[i].decode(), (x1, y1 - 20), 1, 1, classes_font_colors[0], 2, cv2.LINE_AA)

                    cropImage = frame[y1:y2, x1:x2]
                    height, width, channels = cropImage.shape
                    if width > 0 and height > 0:
                        retval, jpgImage = cv2.imencode('.jpg', cropImage)
                        base64Image = base64.b64encode(jpgImage)

                        print("Found {} at keyframe {}: object - {}, prob - {}".format(
                            video_serial, keyframe, self.meta.names[i], dets[j].prob[i]))

                        # benchmark.saveImage(cropImage, self.meta.names[i])  # benchmark

                        # - JSON message to send in callback
                        # - base64 image
                        # - keyframe.toString().padStart(8, 0)
                        # - targetObject and const wrapType = detectedObject.type.replace(' ', '_');
                        # - Prob threshold or detectedObject.percentage.slice(0, -1) > AI.default.threshold
                        dataURL = "data:image/jpeg;base64," + \
                            str(base64Image)  # dataURL scheme
                        bbox = [x1, y1, b.w, b.h]
                        bboxes.append(bbox)
                        confidences.append(dets[j].prob[i])
                        objectTypes.append(objectType)
                        dataURLs.append(dataURL)
                        foundObject = True

        features = self.encoder(frame, bboxes)
        detections = [Detection(bbox, confidence, feature) for bbox,
                      confidence, feature in zip(bboxes, confidences, features)]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        nms_max_overlap = 1.0
        indices = preprocessing.non_max_suppression(
            boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        self.tracker.predict()
        self.tracker.update(detections)

        tracking = []

        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            bbox = track.to_tlwh()
            tracking.append({
                "track_id": str(track.track_id),
                "bbox": {
                    "x": bbox[0],
                    "y": bbox[1],
                    "w": bbox[2],
                    "h": bbox[3],
                },
            })
            # bbox = track.to_tlbr()
            # cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
            # cv2.putText(frame, str(track.track_id),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)

        detectedObjects = []
        for bbox, confidence, objectType, dataURL in zip(bboxes, confidences, objectTypes, dataURLs):
            detectedObjects.append({
                "bbox": {
                    "x": bbox[0],
                    "y": bbox[1],
                    "w": bbox[2],
                    "h": bbox[3],
                },
                "confidence": confidence,
                "objectType": objectType,
                "dataURL": dataURL,
            })

        msg = {
            "type": "DETECTED",
            "robotId": robotId,
            "videoId": videoId,
            "keyframe": keyframe,
            "frame": {
                "width": im.w,
                "height": im.h,
            },
            "detectedObjects": detectedObjects,
            "tracking": tracking,
            "frame_time": time.isoformat(),
            "detect_time": datetime.now().isoformat(),
        }

        self.resultQueue.put([robotId, videoId, msg])

        # cv2.imshow('frame', frame)
        # cv2.waitKey(1)

        if isinstance(arr, bytes):
            free_image(im)
        free_detections(dets, num)

        if isNeedSaveImage and foundObject:
            t = threading.Thread(target=saveImage, args=(filepath, frame))
            t.start()


def array_to_image(arr):
    # need to return old values to avoid python freeing memory
    arr = arr.transpose(2, 0, 1)
    c, h, w = arr.shape[0:3]
    arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
    data = arr.ctypes.data_as(POINTER(c_float))
    im = IMAGE(w, h, c, data)
    return im, arr


def saveImage(filepath, frame):
    cv2.imwrite(filepath, frame)


def initSaveImage():
    if isNeedSaveImage:
        if not os.path.exists(fullImageDir):
            try:
                os.makedirs(fullImageDir)
            except OSError as exc:  # Guard against race condition
                print("OSError:cannot make directory.")
        else:
            fileList = os.listdir(fullImageDir)
            for fileName in fileList:
                os.remove(fullImageDir + "/" + fileName)


darknetWorkers = []


def initDarknetWorkers(numWorkers, numGpus, detectQueue, resultQueue):
    print("darknet numWorkers : {}, numGpus : {}".format(numWorkers, numGpus))

    for i in range(numWorkers):
        worker = Darknet(i, numGpus, detectQueue, resultQueue)
        darknetWorkers.append(worker)
        worker.start()
        print("darknet worker {} started".format(i))


def deinitDarknetWorkers():
    for worker in darknetWorkers:
        print("darknet worker {} stopped".format(worker.index))
        worker.isStop.value = True
    darknetWorkers.clear()


def getDetectRates():
    detectRates = []
    for i in range(len(darknetWorkers)):
        detectRates.append(darknetWorkers[i].detectRate.value)

    return detectRates
