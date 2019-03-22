from ctypes import *
import time
import threading
import multiprocessing as mp
from multiprocessing import Queue, Value
from streamVideo import StreamVideo
import darknet

#stream = "rtsp://admin:12345678@192.168.1.110/ch01.264?ptype=udp"
stream = "rtsp://admin:Obodroid@192.168.1.64"

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
    streamVideo = StreamVideo(stream, '1-1',isStop, captureQueue)
    streamVideo.start()

    streamVideo.join()
    print('after join')
