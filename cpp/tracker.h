#ifndef __TRACKER_H__
#define __TRACKER_H__

#include <opencv2/opencv.hpp>
#include <NvInfer.h>
//#include <torch/script.h>
#include <torch/csrc/api/include/torch/torch.h>
#include <cuda_runtime_api.h>
#include <vector>

//#include <math.h>

//#include "utils.h"

using namespace cv;
using namespace nvinfer1;
using namespace at;
using namespace torch;
using namespace std;

class /*__declspec(dllexport)*/ Tracker
{
private:
    ICudaEngine* mEngine;
    IExecutionContext* mContext;

    // void* m_inputHostBuffer;
    // void* m_inputDeviceBuffer;
    // void* m_outputHostBuffer;
    // void* m_outputDeviceBuffer;

    int outputSize;
    int outputByteSize;

    cudaStream_t m_stream;
    Tensor m_zFeat;
    Tensor m_xFeat;
    vector<void*> mDeviceBindings;

    int center_x, center_y, width, height;
    float scale, m_zSize, m_xSize;
    float m_scales[3];
    float m_penalty;
    // Mat hanming_window;
    Tensor mHannWindow;

    void PreProcess(const Mat& src, Tensor& dst, const Rect2d& roi, int size, int outSize);

public:
    Tracker();
    ~Tracker();
    void Load(const String& fn);
    void Init(const Mat& img, const Rect2d& roi);
    void Update(const Mat& img, Rect2d& roi);
};

#endif