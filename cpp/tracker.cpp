#include "tracker.h"
#include <iostream>
#include <fstream>


using namespace std;
namespace F = torch::nn::functional;


class Logger : public ILogger
{
    void log(Severity severity, const char* msg) override
    {
        //不提示INFO信息，只显示警告和错误
        if (severity != Severity::kINFO)
        {
            std::cout << msg << std::endl;
        }
    }
}gLogger;




Tracker::Tracker()
{
    center_x = 0;
    center_y = 0;
    width = 0;
    height = 0;

    scale = 0.0;
    m_zSize = 0.0;
    m_xSize = 0.0;

    m_scales[0] = 0.964;
    m_scales[1] = 1.0;
    m_scales[2] = 1.0375;

    m_penalty = 0.96;

    // hanming_window = Hanming_weight(272, 272);

    torch::TensorOptions options;
    options= options.dtype(kFloat32).device(kCUDA);
    Tensor hann = torch::hann_window(272, options);
    mHannWindow = torch::ger(hann, hann);
    mHannWindow /= mHannWindow.sum();
    mHannWindow *= 0.176;

    m_zFeat = at::zeros({1 ,256 ,6 , 6}, kFloat32).to(kCUDA);
    m_xFeat = at::zeros({3 ,256 ,22 , 22}, kFloat32).to(kCUDA);

    cudaStreamCreate(&m_stream);

}

Tracker::~Tracker()
{
}


void Tracker::Load(const String& fn)
{
    size_t size{ 0 };
    vector<char> trtModelStream_;
    ifstream file(fn.c_str(), ios::binary);
    if (file.good())
    {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream_.resize(size);
        //cout << "size:" << trtModelStream_.size() << endl;
        file.read(trtModelStream_.data(), size);
        file.close();
    }
    //cout << "size" << size;
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    mEngine = runtime->deserializeCudaEngine(trtModelStream_.data(), size, nullptr);
    mContext = mEngine->createExecutionContext();

    // deal with first run slow issue
    Mat tmp(100, 100, CV_8UC3);
    Rect2d roi(40, 40, 20, 20);
    Init(tmp, roi);
    Update(tmp, roi);
}


void Tracker::Init(const Mat& img, const Rect2d& roi)
{
    // exemplar and search sizes
    float context = (roi.width + roi.height) * .5f;
    m_zSize = sqrt((roi.width + context) * (roi.height + context));
    m_xSize = m_zSize * 2.0f;

    // prepare input data
    Mat z;
    PreProcess(img, z, roi, m_zSize, 127);

    Tensor tz = torch::from_blob(z.data, {1, 127, 127, 3}, torch::kUInt8).to(at::kCUDA).permute({0, 3, 1, 2}).contiguous().to(torch::kFloat32);

    // allocate buffers
    Dims inputDims = mEngine->getProfileDimensions(0, 0, OptProfileSelector::kMIN);
    mContext->setBindingDimensions(0, inputDims);
    Dims outputDims = mContext->getBindingDimensions(1);
 
    mDeviceBindings.clear();
    mDeviceBindings.emplace_back(tz.data_ptr());
    mDeviceBindings.emplace_back(m_zFeat.data_ptr());

    // Asynchronously enqueue the inference work
    mContext->enqueueV2(mDeviceBindings.data(), m_stream, nullptr);
    // Wait for the work in the m_stream to complete
    cudaStreamSynchronize(m_stream);

    inputDims = mEngine->getProfileDimensions(0, 0, OptProfileSelector::kMAX);
    mContext->setBindingDimensions(0, inputDims);
    outputDims = mContext->getBindingDimensions(1);

    mDeviceBindings[1] = m_xFeat.data_ptr();
}


void Tracker::Update(const Mat& img, Rect2d& roi)
{   
    // need speed up //////////////////////////////////////////////////////////////////////////////////////////////////
    Tensor txs[3];
    Mat x;
    for (int i = 0; i < 3; i++) {
        PreProcess(img, x, roi, m_xSize * m_scales[i], 255);
        txs[i] = at::from_blob(x.data, {255, 255, 3}, torch::kUInt8).to(at::kCUDA);
    }
    Tensor tx = stack({txs[0], txs[1], txs[2]}).permute({0, 3, 1, 2}).contiguous().to(torch::kFloat32);
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    mDeviceBindings[0] = tx.data_ptr();
    
    TickMeter tm;
    tm.start();
    // Asynchronously enqueue the inference work
    mContext->enqueueV2(mDeviceBindings.data(), m_stream, nullptr);
    // Wait for the work in the m_stream to complete
    cudaStreamSynchronize(m_stream);
    tm.stop();
    cout << tm.getTimeMilli() << endl;
    
    // cross correlation
    Tensor response = F::conv2d(m_xFeat, m_zFeat);

    // penalize scale changes
    response[0] *= m_penalty;
    response[2] *= m_penalty;

    // need speed up //////////////////////////////////////////////////////////////////////////////////////////////////
    // peak scale
    float maxRec = INT_MIN;
    int scaleId = 0;
    for(int i=0; i<3; i++){
        float maxV = response[i].max().item().to<float>();
        if(maxRec < maxV){
            maxRec = maxV;
            scaleId = i;
        }
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    // upsample
    Tensor response1 = response[scaleId].unsqueeze(0);
    response1 = F::interpolate(response1, F::InterpolateFuncOptions().size(vector<int64_t>{272, 272}).mode(torch::kBicubic).align_corners(false)).squeeze();

    // peak location
    response1 = response1 * 0.824f + mHannWindow;
    int maxIdx = response1.argmax().item().to<int>();
    int maxIdxY = maxIdx / 272;
    int maxIdxX = maxIdx % 272;
    float dispx = maxIdxX - 271/2.0;
    float dispy = maxIdxY - 271/2.0;

    // update roi
    dispx /= 2.0;
    dispy /= 2.0;
    dispx = dispx * m_xSize / 255;
    dispy = dispy * m_xSize / 255;
    
    m_xSize *= m_scales[scaleId];

    roi.x += dispx;
    roi.y += dispy;
    roi.width *= m_scales[scaleId];
    roi.height *= m_scales[scaleId];
}


void Tracker::PreProcess(const Mat& src, Mat& dst, const Rect2d& roi, int size, int outSize)
{
    // half
    int hw = roi.width / 2;
    int hh = roi.height / 2;
    int hs = size / 2;

    // roi center
    int cx = roi.x + hw;
    int cy = roi.y + hh;

    // new roi
    Rect newRoi(cx-hs, cy-hs, size, size);

    // left and top margin
    int left = max(0, hs - cx);
    int top = max(0, hs - cy);

    // intersection of new roi and src
    newRoi &= Rect(0, 0, src.cols, src.rows);

    // right and down margin
    int right = size - newRoi.width - left;
    int bottom = size - newRoi.height - top;
    
    // crop and pad
    src(newRoi).copyTo(dst);
    copyMakeBorder(dst, dst, top, bottom, left, right, cv::BORDER_REPLICATE);

    // resize
    cv::resize(dst, dst, Size(outSize, outSize));
}