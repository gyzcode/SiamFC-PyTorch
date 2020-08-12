#include <opencv2/opencv.hpp>
#include "cv_ui.h"
#include "tracker.h"

using namespace cv;

int main()
{
    CvUI ui;

    Tracker myTracker;
    myTracker.Load("/home/gyz/workzone/siamfc-pytorch/pretrained/siamfc_alexnet_pruning_e50_dynamic_int8.engine");

    namedWindow("display");
    setMouseCallback("display", ui.OnMouse);

    VideoCapture cap(0);
    Mat frame;
    Rect2d roi;
    bool running = true;
    bool tracking = false;
    while (running) {
        cap >> frame;
        if(!frame.empty()){
            if(ui.newInit){
                ui.newInit = false;
                roi = ui.GetBb();
                myTracker.Init(frame, roi);
                tracking = true;
            }

            if (ui.mode == SELECT){
                rectangle(frame, ui.GetTl(), ui.GetBr(), CV_RGB(0, 0, 255), 2);
            }

            if(tracking){
                TickMeter tm;
                tm.start();
                myTracker.Update(frame, roi);
                tm.stop();
                cout << tm.getTimeMilli() << endl;
                rectangle(frame, roi, CV_RGB(255, 0, 0), 2);
            }
            
            imshow("display", frame);
            int key = waitKey(1);
            if(key == 27){
                running = false;
            }
        }
    }
    
    cap.release();

    return 0;
}