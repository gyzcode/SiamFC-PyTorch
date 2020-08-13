#include "tracker.h"
#include <opencv2/opencv.hpp>

using namespace cv;

int main()
{
    Tracker myTracker;
    myTracker.Load("/home/gyz/workzone/siamfc-pytorch/pretrained/siamfc_alexnet_pruning_e50_x_int8.engine", 'x');
    myTracker.Load("/home/gyz/workzone/siamfc-pytorch/pretrained/siamfc_alexnet_pruning_e50_z_int8.engine", 'z');
    Mat frame;
    String fn;
    float timeCost = 0;
    Rect2d roi(204,150,17,50);  //Crossing
    int numFrame = 120;
    //Rect2d roi(288,143,35,42);  //Boy
    //int numFrame = 602;
    for(int i=1; i<=numFrame; i++) {
        fn = format("/home/gyz/dataset/otb100/Crossing/img/%04d.jpg", i);
        frame = imread(fn);

        TickMeter tm;
        tm.start();
        if(i == 1) {
            myTracker.Init(frame, roi);
        }
        else {
            myTracker.Update(frame, roi);
        }
        tm.stop();
        timeCost += tm.getTimeMilli();

        rectangle(frame, roi, CV_RGB(255, 0, 0), 2);
        imshow("Display", frame);
        waitKey(1);
    }
    cout << timeCost / numFrame << endl;

    destroyAllWindows();


    return 0;
}
