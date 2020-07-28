#ifndef __CV_UI_H__
#define __CV_UI_H__

#include <opencv2/opencv.hpp>

using namespace cv;

enum MODE {INIT, SELECT};

class CvUI
{
private:
    static MODE mode;
    static Point tl;
    static Point br;
    static bool newInit;
    Point get_tl();
public:
    CvUI();
    ~CvUI();
    static void OnMouse(int event, int x, int y, int flags, void* ustc);


};

#endif