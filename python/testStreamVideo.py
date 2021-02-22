from ctypes import *
import time
import threading
import multiprocessing as mp
from multiprocessing import Value ,Queue
import queue
from streamVideo import StreamVideo
import darknet

stream = "rtsp://admin:admin@192.168.200.104:554/1"

def stop(sv):
    print('start function')
    time.sleep(10)
    sv.stop()
    print('end function')

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    # darknet.initDarknetWorkers(1, 4)

    isStop = Value(c_bool, False)
    dropFrameCount = Value('i', 0)
    detectQueue = Queue(maxsize=10)
    video_serial, keyframe, frame, time = detectQueue.get(timeout=0.1)
    detectThroughput = Value('i', 0)
    streamVideo = StreamVideo(stream, '1-1', isStop, True, dropFrameCount, detectQueue, detectThroughput)
    streamVideo.start()

    t = threading.Thread(target=stop, args=(streamVideo,))
    #t.start()

    streamVideo.join()
