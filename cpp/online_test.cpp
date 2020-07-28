#include <opencv2/opencv.hpp>
#include "cv_ui.h"

using namespace cv;

int main()
{
    CvUI ui;

    namedWindow("display");
    setMouseCallback("display", ui.OnMouse);

    VideoCapture cap(0);
    Mat frame;
    while (1) {
        cap >> frame;
        if(!frame.empty()){
            imshow("display", frame);
            waitKey(1);
        }
    }
    
    cap.release();

    return 0;
}