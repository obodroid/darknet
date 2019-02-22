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
    # ctx = mp.get_context('spawn')
    
    # darknet.initDarknetWorkers(1, 4)

    captureQueue = Queue(maxsize=10)
    isStop = Value(c_bool, False)
    streamVideo = StreamVideo(stream, '1-1', captureQueue, isStop)
    streamVideo.start()

    t = threading.Thread(target=stop, args=(streamVideo,))
    t.start()

    streamVideo.join()
    print('after join')
