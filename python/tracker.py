from ctypes import *
from multiprocessing import Process
import setproctitle
import numpy as np
import cv2
import os
import sys

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet

imgEncPath = b"/src/data/deep_sort/mars-small128.pb"


class DeepSort(Process):
    def __init__(self, video_serial, trackingQueue, resultQueue):
        Process.__init__(self)
        self.daemon = True
        self.encoder = None
        self.tracker = None
        self.video_serial = video_serial
        self.trackingQueue = trackingQueue
        self.resultQueue = resultQueue

    def run(self):
        setproctitle.setproctitle("Tracker {}".format(self.video_serial))
        print('Tracker {}'.format(self.video_serial))

        max_cosine_distance = 0.3
        nn_budget = None

        self.encoder = gdet.create_box_encoder(imgEncPath, batch_size=1)
        metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric)

        while True:
            while not self.trackingQueue.empty():
                robotId, videoId, msg, frame, bboxes, confidences = self.trackingQueue.get()
                video_serial = robotId + "-" + videoId
                print('Track {}'.format(video_serial))

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
                msg['detectedObjects'] = [msg['detectedObjects'][i] for i in indices]

                # Call the tracker
                self.tracker.predict()
                self.tracker.update(detections)

                for detection_id, detectedObject in zip(indices, msg['detectedObjects']):
                    for track in self.tracker.tracks:
                        if not track.is_confirmed() or track.time_since_update > 1:
                            continue

                        if track.detection_id == detection_id:
                            detectedObject["track_id"] = str(track.track_id)
                            tracking_bbox = track.to_tlwh()
                            detectedObject["tracking_bbox"] = {
                                "x": tracking_bbox[0],
                                "y": tracking_bbox[1],
                                "w": tracking_bbox[2],
                                "h": tracking_bbox[3],
                            },
                            # bbox = track.to_tlbr()
                            # cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
                            # cv2.putText(frame, str(track.track_id),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)
                            break

                self.resultQueue.put([robotId, videoId, msg])
            cv2.waitKey(1)
            sys.stdout.flush()