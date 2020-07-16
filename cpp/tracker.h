#ifndef __TRACKER_H__
#define __TRACKER_H__

#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <torch/script.h>

//#include <math.h>
//#include <vector>
//#include "utils.h"

using namespace cv;
using namespace nvinfer1;

class /*__declspec(dllexport)*/ Tracker
{
private:
    ICudaEngine* mEngine;
    IExecutionContext* mContext;

    void* m_inputHostBuffer;
    void* m_inputDeviceBuffer;
    void* m_outputHostBuffer;
    void* m_outputDeviceBuffer;

    int center_x, center_y, width, height;
    float scale, target_sz, search_sz;
    float mul_scale[3];
    float mul_penalty[3];
    Mat hanming_window;
    //at::Tensor tmplate;

public:
    Tracker();
    ~Tracker();
    void Load(const String& fn);
    void Init(const Mat& img, const Rect2d& roi);
    void Update(const Mat& img, Rect2d& roi);
};

#endif