#include "tracker.h"
#include <opencv2/opencv.hpp>

using namespace cv;

int main()
{
    Tracker myTracker;
    myTracker.Load("/home/gyz/workzone/siamfc-pytorch/pretrained/siamfc_alexnet_e50_dynamic.engine");
    Mat frame = imread("/home/gyz/Pictures/person.jpg");
    Rect roi(100, 100, 50, 50);
    myTracker.Init(frame, roi);


    return 0;
}