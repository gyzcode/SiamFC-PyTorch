#include "tracker.h"
#include <opencv2/opencv.hpp>

using namespace cv;

int main()
{
    Tracker myTracker;
    myTracker.Load("/home/gyz/workzone/siamfc-pytorch/pretrained/siamfc_alexnet_e50_dynamic.engine");
    Mat frame;
    String fn;
    Rect2d roi(204,150,17,50);
    for(int i=1; i<=120; i++) {
        fn = format("/home/gyz/dataset/otb100/Crossing/img/%04d.jpg", i);
        frame = imread(fn);
        //cvtColor(frame, frame, COLOR_BGR2RGB);

        // TickMeter tm;
        // tm.start();
        if(i == 1) {
            myTracker.Init(frame, roi);
        }
        else {
            myTracker.Update(frame, roi);
        }
        // tm.stop();
        // cout << tm.getTimeMilli() << endl;

        rectangle(frame, roi, CV_RGB(255, 0, 0), 2);
        imshow("Display", frame);
        waitKey(1);
    }

    destroyAllWindows();


    return 0;
}