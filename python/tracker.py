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
    def __init__(self, video_serial, isStop, gpuIndex, trackingQueue, resultQueue):
        Process.__init__(self)
        self.daemon = True
        self.encoder = None
        self.tracker = None
        self.isStop = isStop
        self.isDisplay = False
        self.gpuIndex = gpuIndex
        self.video_serial = video_serial
        self.trackingQueue = trackingQueue
        self.resultQueue = resultQueue

    def run(self):
        setproctitle.setproctitle("Tracker {}".format(self.video_serial))
        print('Tracker {}'.format(self.video_serial))

        max_cosine_distance = 0.45
        nn_budget = 100

        self.encoder = gdet.create_box_encoder(imgEncPath, batch_size=1, gpu_index=self.gpuIndex)
        metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric, max_iou_distance=0.7, max_age=50, n_init=5)

        while self.isStop.value is False:
            while not self.trackingQueue.empty():
                robotId, videoId, msg, frame, bboxes, confidences, objectTypes = self.trackingQueue.get()
                video_serial = robotId + "-" + videoId
                print('Tracker {} at keyframe {}'.format(video_serial, msg['keyframe']))

                features = self.encoder(frame, bboxes)
                detections = [Detection(bbox, confidence, feature, objectType) 
                    for bbox, confidence, feature, objectType 
                    in zip(bboxes, confidences, features, objectTypes)]

                # Run non-maxima suppression.
                boxes = np.array([d.tlwh for d in detections])
                scores = np.array([d.confidence for d in detections])
                nms_max_overlap = 1.0
                indices = preprocessing.non_max_suppression(
                    boxes, nms_max_overlap, scores)

                detections = [detections[i] for i in indices]
                msg['detectedObjects'] = [msg['detectedObjects'][i] for i in indices]
                print("detections: {}".format([d.objectType for d in detections]))

                # Call the tracker
                self.tracker.predict()
                self.tracker.update(detections)

                for detection_id, detectedObject in zip(np.arange(len(msg['detectedObjects'])), msg['detectedObjects']):
                    if detectedObject["confidence"] < 0.8:
                        continue
                        
                    for track in self.tracker.tracks:
                        if not track.is_confirmed() or track.time_since_update > 0:
                            print("Tracker {} at keyframe {} track {} missed x {} y {}".format( \
                                self.video_serial, msg['keyframe'], str(track.track_id), int(track.to_tlwh()[0]), int(track.to_tlwh()[1])))
                            bbox = track.to_tlbr()
                            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(127,127,127), 2)
                            cv2.putText(frame, "{}".format(str(track.track_id)),(int(bbox[0]), int(bbox[1]) - 20), 0, 5e-3 * 100, (0,127,0), 2)
                            continue

                        if track.detection_id == detection_id:
                            print("Tracker {} at keyframe {} track {} {} x {} y {}".format( \
                                self.video_serial, msg['keyframe'], detectedObject["objectType"] \
                                , str(track.track_id), int(track.to_tlwh()[0]), int(track.to_tlwh()[1])))

                            detectedObject["track_id"] = str(track.track_id)
                            tracking_bbox = track.to_tlwh()
                            detectedObject["tracking_bbox"] = {
                                "x": tracking_bbox[0],
                                "y": tracking_bbox[1],
                                "w": tracking_bbox[2],
                                "h": tracking_bbox[3],
                            }

                            if self.isDisplay:
                                bbox = track.to_tlbr()
                                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
                                cv2.putText(frame, "{} {}".format(detectedObject["objectType"], \
                                    str(track.track_id)),(int(bbox[0]), int(bbox[1]) - 20), 0, 5e-3 * 100, (0,255,0), 2)
                            break

                self.resultQueue.put([robotId, videoId, msg])

                if self.isDisplay:
                    print("Tracker {} show frame".format(self.video_serial))
                    title = "track : {}".format(self.video_serial)
                    cv2.putText(frame, "keyframe {}".format(msg['keyframe']),(30, 100), 0, 5e-3 * 100, (0,0,255), 2)
                    cv2.imshow(title, frame)
                    cv2.waitKey(1)

                cv2.waitKey(1)
                sys.stdout.flush()

        cv2.waitKey(1)
        sys.stdout.flush()
        
        print("Tracker {} Stopped".format(self.video_serial))
