from ctypes import *
import time
import threading
import multiprocessing as mp
from multiprocessing import Queue, Value
from streamVideo import StreamVideo
import darknet

stream = "rtsp://admin:12345678@192.168.1.101/ch01.264?ptype=udp"

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
    detectThroughput = Value('i', 0)
    streamVideo = StreamVideo(stream, '1-1', isStop, True, dropFrameCount, detectQueue, detectThroughput)
    streamVideo.start()

    t = threading.Thread(target=stop, args=(streamVideo,))
    t.start()

    streamVideo.join()
