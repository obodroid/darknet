#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include "test.h"

int main(int argc, char* argv[])
{
    int pos_frames;
    char* filename = argv[1];
    printf("video file: %s\n", filename);

    clock_t start_clock = clock();
    struct timeval tv = {0};
    gettimeofday(&tv, NULL);
    double start_time = tv.tv_sec + ((double)tv.tv_usec / 1000000);

    CvCapture* cap = cvCaptureFromFile(filename);
    printf("capture from file (clock): %f seconds\n", (double)(clock() - start_clock) / CLOCKS_PER_SEC);

    int w = cvGetCaptureProperty(cap, CV_CAP_PROP_FRAME_WIDTH);
    int h = cvGetCaptureProperty(cap, CV_CAP_PROP_FRAME_HEIGHT);
    int frame_rate = cvGetCaptureProperty(cap, CV_CAP_PROP_FPS);
    // int buffer_size = cvGetCaptureProperty(cap, CV_CAP_PROP_BUFFERSIZE);

    printf("w: %d, h: %d, frame rate: %d\n", w, h, frame_rate);
    // printf("buffer size: %d\n", buffer_size);
    
    IplImage *frame;
    while(1)
    {
        pos_frames = cvGetCaptureProperty(cap, CV_CAP_PROP_POS_FRAMES);
        printf("frame index: %d\n", pos_frames);
        
        start_clock = clock();
        gettimeofday(&tv, NULL);
        double current_time = tv.tv_sec + ((double)tv.tv_usec / 1000000);

        frame = cvQueryFrame(cap);

        printf("query frame %d (clock): %f seconds\n", pos_frames, (double)(clock() - start_clock) / CLOCKS_PER_SEC);
        printf("latency from start (real time): %f seconds\n", current_time - start_time);

        if(!frame) break;
        cvShowImage("video", frame);
        char c = cvWaitKey(1);
        if(c == 27) break;
    }
    
    return 0;
}
