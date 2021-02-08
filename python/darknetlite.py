# from ctypes import *
# import cv2
# import numpy as np
# import threading
# from multiprocessing import Process, Value
# from threading import Timer
# import setproctitle
# import os
# import sys
# from datetime import datetime
# import time
# import base64
# import random
# import math

from ctypes import *
import math
import random
import os

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
                ("sort_class", c_int),
                ("uc", POINTER(c_float)),
                ("points", c_int)]

class DETNUMPAIR(Structure):
    _fields_ = [("num", c_int),
                ("dets", POINTER(DETECTION))]

class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]


class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

def network_width(net):
    return lib.network_width(net)


def network_height(net):
    return lib.network_height(net)


def bbox2points(bbox):
    """
    From bounding box yolo format
    to corner points cv2 rectangle
    """
    x, y, w, h = bbox
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def class_colors(names):
    """
    Create a dict with one random BGR color for each
    class name
    """
    return {name: (
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255)) for name in names}


def load_network(config_file, data_file, weights, batch_size=1):
    """
    load model description and weights from config files
    args:
        config_file (str): path to .cfg model file
        data_file (str): path to .data model file
        weights (str): path to weights
    returns:
        network: trained model
        class_names
        class_colors
    """
    network = load_net_custom(
        config_file.encode("ascii"),
        weights.encode("ascii"), 0, batch_size)
    metadata = load_meta(data_file.encode("ascii"))
    class_names = [metadata.names[i].decode("ascii")for i in range(metadata.classes)]
    print(class_names)
    colors = class_colors(class_names)
    return network, class_names, colors


def print_detections(detections, coordinates=False):
    print("\nObjects:")
    for label, confidence, bbox in detections:
        x, y, w, h = bbox
        if coordinates:
            print("{}: {}%    (left_x: {:.0f}   top_y:  {:.0f}   width:   {:.0f}   height:  {:.0f})".format(label, confidence, x, y, w, h))
        else:
            print("{}: {}%".format(label, confidence))


def draw_boxes(detections, image, colors):
    import cv2
    for label, confidence, bbox in detections:
        left, top, right, bottom = bbox2points(bbox)
        cv2.rectangle(image, (left, top), (right, bottom), colors[label], 1)
        cv2.putText(image, "{} [{:.2f}]".format(label, float(confidence)),
                    (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    colors[label], 2)
    return image

def decode_detection(detections):
    decoded = []
    for label, confidence, bbox in detections:
        confidence = str(round(confidence * 100, 2))
        decoded.append((str(label), confidence, bbox))
    return decoded


def remove_negatives(detections, class_names, num):
    """
    Remove all classes with 0% confidence within the detection
    """
    predictions = []
    for j in range(num):
        print(j)
        print("classes",detections[j].classes)
        for idx, name in enumerate(class_names):
            if detections[j].prob[idx] > 0:
                bbox = detections[j].bbox
                bbox = (bbox.x, bbox.y, bbox.w, bbox.h)
                x1,y1,x2,y2=bbox2points(bbox)
                predictions.append((name, detections[j].prob[idx], (bbox)))
                print("predictions",predictions)
    return predictions ,x1,y1,x2,y2

def nnDetect(self, video_serial, keyframe, frame, time,class_names,network,image):
        #self.detectCount += 1
        # print("darknet {} nnDetect {}, keyframe {}".format(
        #     self.index, video_serial, keyframe))
        # robotId, videoId = video_serial.split('-')
        # red for palmup --> stop, green for thumbsup --> go
    classes_box_colors = [(0, 0, 255), (0, 255, 0)]
    classes_font_colors = [(255, 255, 0), (0, 255, 255)]
    foundObject = False
        # filename = '{}_{}.jpg'.format(video_serial, keyframe)
        # filepath = '{}/{}'.format(fullImageDir, filename)
    bboxes = []
    confidences = []
    objectTypes = []
    dataURLs = []
        #for test image detect only
        #frame=cv2.imread
        # if True:
        #     displayFrame = image.copy()
        
    pnum = pointer(c_int(0))
    predict_image(network, image)
    detections = get_network_boxes(network, image.w, image.h,
                                   thresh, hier_thresh, None, 0, pnum, 0)
    num = pnum[0]
    if nms:
        do_nms_sort(detections, num, len(class_names), nms)

    print(range(num))
        
    predictions,x1,y1,x2,y2= remove_negatives(detections, class_names, num)
    predictions = decode_detection(predictions)
    free_detections(detections, num)

        
    return sorted(predictions, key=lambda x: x[1])

def detect_image(network, class_names, image, thresh=.5, hier_thresh=.5, nms=.45):
        # """
        # Returns a list with highest confidence class and their bbox
        # """
    pnum = pointer(c_int(0))
    predict_image(network, image)
    detections = get_network_boxes(network, image.w, image.h,
                                   thresh, hier_thresh, None, 0, pnum, 0)
    num = pnum[0]
    if nms:
        do_nms_sort(detections, num, len(class_names), nms)
    predictions = remove_negatives(detections, class_names, num)
    predictions = decode_detection(predictions)
    free_detections(detections, num)
    return sorted(predictions, key=lambda x: x[1])

# class Darknet(Process):
#     def __init__(self, index, numGpus, threshold, detectQueue, resultQueue):
#         Process.__init__(self)
#         self.daemon = True
#         self.net = None
#         self.meta = None
#         self.index = index
#         self.numGpus = numGpus
#         self.threshold = threshold
#         self.detectCount = 0
#         self.detectRate = Value('i', -1)
#         self.detectQueue = detectQueue
#         self.resultQueue = resultQueue
#         self.isStop = Value(c_bool, False)
#         self.isDisplay = True


#     def run(self):
#         setproctitle.setproctitle("Darknet {}".format(self.index))
#         gpuIndex = (self.index % self.numGpus) + \
#             ((int(os.environ['CONTAINER_INDEX']) - 1) * self.numGpus) + 1 if 'CONTAINER_INDEX' in os.environ else 0
#         set_gpu(gpuIndex)
#         print("Load darknet worker = {} with gpuIndex = {}".format(
#             self.index, gpuIndex))
#         self.net,class_names,class_colors = load_network(configPath,metaPath,weightPath,1)
#         print("1")
#         print("darknet {} initialized".format(self.index))

#         self.monitorDetectRate()

#         while not self.isStop.value:
#             try:
#                 video_serial, keyframe, frame, time = self.detectQueue.get(
#                     timeout=0.1)
#                 self.nnDetect(video_serial, keyframe, frame, time,class_names)
#             except Exception:
#                 pass
#             sys.stdout.flush()


#     def monitorDetectRate(self):
#         t = threading.Timer(1.0, self.monitorDetectRate)
#         t.setDaemon(True)
#         t.start()
#         self.detectRate.value = self.detectCount
#         self.detectCount = 0

#     def detect_image(network, class_names, image, thresh=.5, hier_thresh=.5, nms=.45):
#         # """
#         # Returns a list with highest confidence class and their bbox
#         # """
#         pnum = pointer(c_int(0))
#         predict_image(network, image)
#         detections = get_network_boxes(network, image.w, image.h,
#                                    thresh, hier_thresh, None, 0, pnum, 0)
#         num = pnum[0]
#         if nms:
#             do_nms_sort(detections, num, len(class_names), nms)
#         predictions = remove_negatives(detections, class_names, num)
#         predictions = decode_detection(predictions)
#         free_detections(detections, num)
#         return sorted(predictions, key=lambda x: x[1])

#     def nnDetect(self, video_serial, keyframe, frame, time,class_names,network,image):
#         #self.detectCount += 1
#         print("test")
#         # print("darknet {} nnDetect {}, keyframe {}".format(
#         #     self.index, video_serial, keyframe))
#         # robotId, videoId = video_serial.split('-')
#         # red for palmup --> stop, green for thumbsup --> go
#         classes_box_colors = [(0, 0, 255), (0, 255, 0)]
#         classes_font_colors = [(255, 255, 0), (0, 255, 255)]
#         foundObject = False
#         # filename = '{}_{}.jpg'.format(video_serial, keyframe)
#         # filepath = '{}/{}'.format(fullImageDir, filename)
#         bboxes = []
#         confidences = []
#         objectTypes = []
#         dataURLs = []
#         #for test image detect only
#         #frame=cv2.imread
#         # if True:
#         #     displayFrame = image.copy()
        
#         pnum = pointer(c_int(0))
#         predict_image(network, image)
#         detections = get_network_boxes(network, image.w, image.h,
#                                    thresh, hier_thresh, None, 0, pnum, 0)
#         num = pnum[0]
#         if nms:
#             do_nms_sort(detections, num, len(class_names), nms)

#         print(range(num))
        
#         predictions,x1,y1,x2,y2= remove_negatives(detections, class_names, num)
#         predictions = decode_detection(predictions)

#         # if True:
#         #     cv2.rectangle(displayFrame, (x1, y1), (x2, y2), classes_box_colors[1], 2)
#         #     cv2.putText(displayFrame, class_names, (x1, y1 - 20), 1, 1, classes_font_colors[0], 2, cv2.LINE_AA)
#         #     cv2.waitKey(1)

#         #     cropImage = frame[y1:y2, x1:x2]
#         #     height, width, channels = cropImage.shape
#         #     if width > 0 and height > 0:
#         #         print("if width,height>0")
#         #         retval, jpgImage = cv2.imencode('.jpg', cropImage)
#         #         base64Image = base64.b64encode(jpgImage)
#         #         print("Found {} at keyframe {}: object - {}, prob - {}, x - {}, y - {}".format(
#         #          video_serial, keyframe, objectType, detections[j].prob[i], x1, y1))
#         #         dataURL = "data:image/jpeg;base64," + str(base64Image.decode())
#         #         bbox = [x1, y1, b.w, b.h]
#         #         bboxes.append(bbox)
#         #         confidences.append(detections[j].prob[idx])
#         #         objectTypes.append(objectType)
#         #         dataURLs.append(dataURL)
#         #         foundObject = True
#         # detectedObjects = []
#         # for bbox, confidence, objectType, dataURL in zip(bboxes, confidences, objectTypes, dataURLs):
#         #     detectedObject = {
#         #         "bbox": {
#         #             "x": bbox[0],
#         #             "y": bbox[1],
#         #             "w": bbox[2],
#         #             "h": bbox[3],
#         #         },
#         #         "confidence": confidence,
#         #         "objectType": objectType,
#         #         "dataURL": dataURL,
#         #     }
#         #     detectedObjects.append(detectedObject)
#         # # print("detectedObjects",detectedObjects)
#         # msg = {
#         #     "type": "DETECTED",
#         #     "robotId": robotId,
#         #     "videoId": videoId,
#         #     "keyframe": keyframe,
#         #     "frame": {
#         #         "width": im.w,
#         #         "height": im.h,
#         #     },
#         #     "detectedObjects": detectedObjects,
#         #     "frame_time": time.isoformat(),
#         #     "detect_time": datetime.now().isoformat(),
#         # }

#         # self.resultQueue.put([robotId, videoId, msg, frame, bboxes, confidences, objectTypes])
#         # if self.isDisplay:
#         #     print("Darknet {} show frame".format(video_serial))
#         #     title = "detect : {}".format(video_serial)
#         #     cv2.putText(displayFrame, "keyframe {}".format(keyframe),(30, 70), 0, 5e-3 * 100, (0,0,255), 2)
#         #     cv2.imshow(title, displayFrame)
#         #     cv2.waitKey(1)

#         free_detections(detections, num)

#         # if isNeedSaveImage and foundObject:
#         #     t = threading.Thread(target=saveImage, args=(filepath, frame))
#         #     t.start()
#         return sorted(predictions, key=lambda x: x[1])

# def array_to_image(arr):
#     # need to return old values to avoid python freeing memory
#     arr = arr.transpose(2, 0, 1)
#     c, h, w = arr.shape[0:3]
#     arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
#     data = arr.ctypes.data_as(POINTER(c_float))
#     im = IMAGE(w, h, c, data)
#     return im, arr


# def saveImage(filepath, frame):
#     cv2.imwrite(filepath, frame)


# def initSaveImage():
#     if isNeedSaveImage:
#         if not os.path.exists(fullImageDir):
#             try:
#                 os.makedirs(fullImageDir)
#             except OSError as exc:  # Guard against race condition
#                 print("OSError:cannot make directory.")
#         else:
#             fileList = os.listdir(fullImageDir)
#             for fileName in fileList:
#                 os.remove(fullImageDir + "/" + fileName)


# darknetWorkers = []


# def initDarknetWorkers(numWorkers, numGpus, threshold, detectQueue, resultQueue):
#     print("darknet numWorkers : {}, numGpus : {}".format(numWorkers, numGpus))

#     for i in range(numWorkers):
#         worker = Darknet(i, numGpus, threshold, detectQueue, resultQueue)
#         darknetWorkers.append(worker)
#         worker.start()
#         print("darknet worker {} started".format(i))


# def deinitDarknetWorkers():
#     for worker in darknetWorkers:
#         print("darknet worker {} stopped".format(worker.index))
#         worker.isStop.value = True
#     darknetWorkers.clear()


# def getDetectRates():
#     detectRates = []
#     for i in range(len(darknetWorkers)):
#         detectRates.append(darknetWorkers[i].detectRate.value)

#     return detectRates





#for lib configure
hasGPU = True
if os.name == "nt":
    cwd = os.path.dirname(__file__)
    os.environ['PATH'] = cwd + ';' + os.environ['PATH']
    winGPUdll = os.path.join(cwd, "yolo_cpp_dll.dll")
    winNoGPUdll = os.path.join(cwd, "yolo_cpp_dll_nogpu.dll")
    envKeys = list()
    for k, v in os.environ.items():
        envKeys.append(k)
    try:
        try:
            tmp = os.environ["FORCE_CPU"].lower()
            if tmp in ["1", "true", "yes", "on"]:
                raise ValueError("ForceCPU")
            else:
                print("Flag value {} not forcing CPU mode".format(tmp))
        except KeyError:
            # We never set the flag
            if 'CUDA_VISIBLE_DEVICES' in envKeys:
                if int(os.environ['CUDA_VISIBLE_DEVICES']) < 0:
                    raise ValueError("ForceCPU")
            try:
                global DARKNET_FORCE_CPU
                if DARKNET_FORCE_CPU:
                    raise ValueError("ForceCPU")
            except NameError as cpu_error:
                print(cpu_error)
        if not os.path.exists(winGPUdll):
            raise ValueError("NoDLL")
        lib = CDLL(winGPUdll, RTLD_GLOBAL)
    except (KeyError, ValueError):
        hasGPU = False
        if os.path.exists(winNoGPUdll):
            lib = CDLL(winNoGPUdll, RTLD_GLOBAL)
            print("Notice: CPU-only mode")
        else:
            # Try the other way, in case no_gpu was compile but not renamed
            lib = CDLL(winGPUdll, RTLD_GLOBAL)
            print("Environment variables indicated a CPU run, but we didn't find {}. Trying a GPU run anyway.".format(winNoGPUdll))
else:
    lib = CDLL("/src/darknet/libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

copy_image_from_bytes = lib.copy_image_from_bytes
copy_image_from_bytes.argtypes = [IMAGE,c_char_p]

predict = lib.network_predict_ptr
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

if hasGPU:
    set_gpu = lib.cuda_set_device
    set_gpu.argtypes = [c_int]

init_cpu = lib.init_cpu

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int), c_int]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_batch_detections = lib.free_batch_detections
free_batch_detections.argtypes = [POINTER(DETNUMPAIR), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict_ptr
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

load_net_custom = lib.load_network_custom
load_net_custom.argtypes = [c_char_p, c_char_p, c_int, c_int]
load_net_custom.restype = c_void_p

free_network_ptr = lib.free_network_ptr
free_network_ptr.argtypes = [c_void_p]
free_network_ptr.restype = c_void_p

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

predict_image_letterbox = lib.network_predict_image_letterbox
predict_image_letterbox.argtypes = [c_void_p, IMAGE]
predict_image_letterbox.restype = POINTER(c_float)

network_predict_batch = lib.network_predict_batch
network_predict_batch.argtypes = [c_void_p, IMAGE, c_int, c_int, c_int,
                                   c_float, c_float, POINTER(c_int), c_int, c_int]
network_predict_batch.restype = POINTER(DETNUMPAIR)