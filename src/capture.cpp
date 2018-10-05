#include <iostream> // for standard I/O
#include <opencv2/highgui/highgui.hpp>  // OpenCV window I/O

using namespace std;
using namespace cv;

extern "C" VideoCapture cvCreateStreamCapture(const char* stream_name)
{
    VideoCapture cap(stream_name);
    return cap;
}

extern "C" void print_stream_name(const char* stream_name)
{
    printf("stream name: %s\n", stream_name);
}

extern "C" void run_stream(const char* stream_name) {
    VideoCapture cap = cvCreateStreamCapture(stream_name);
    
    char c;
    int frameNum = -1;          // Frame counter

    if (!cap.isOpened())
    {
        cout  << "Could not open stream " << stream_name << endl;
        return;
    }

    const char* WIN_STREAM = "Stream";

    namedWindow(WIN_STREAM, CV_WINDOW_AUTOSIZE);
    cvMoveWindow(WIN_STREAM, 400, 0);

    Mat streamFrame;

    for(;;)
    {
        cap >> streamFrame;

        if (streamFrame.empty())
        {
            cout << " < < <  END  > > > ";
            break;
        }

        ++frameNum;
        cout << "Frame: #" << frameNum << "\n";

        imshow(WIN_STREAM, streamFrame);

        c = (char)cvWaitKey(1);
        if (c == 27) break;
    }
}