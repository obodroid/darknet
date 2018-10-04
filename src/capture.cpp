#include <stdio.h>
#include "darknet.h"

#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/imgproc/imgproc.hpp>  // Gaussian Blur
#include <opencv2/highgui/highgui.hpp>  // OpenCV window I/O

extern "C" cv::VideoCapture cvCreateStreamCapture(const char* stream_name)
{
    cv::VideoCapture cap(stream_name);
    return cap;
}

extern "C" void print_stream_name(const char* stream_name)
{
    printf("stream name: %s\n", stream_name);
}