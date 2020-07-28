#ifndef __CV_UI_H__
#define __CV_UI_H__

#include <opencv2/opencv.hpp>

enum MODE {INIT, SELECT};

class CvUI
{
private:

public:
    static MODE mode;
    static cv::Point tl;
    static cv::Point br;
    static bool newInit;

    CvUI();
    ~CvUI();
    static void OnMouse(int event, int x, int y, int flags, void* ustc);
    cv::Point GetTl();
    cv::Point GetBr();
    cv::Rect2d GetBb();
};

#endif