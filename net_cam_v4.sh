#rm test_dnn_out.avi

./darknet detector demo ./cfg/coco.data ./cfg/yolov4.cfg ./cfg/yolov4.weights rtsp://admin:admin@192.168.101.13 -i 0 -thresh 0.25



