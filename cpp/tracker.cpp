#include "tracker.h"
#include <iostream>
#include <fstream>
//#include <cuda_runtime.h>

using namespace std;


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


Mat Gaussion_weight(int height, int width, float theta);
Mat Hanming_weight(int height, int width);


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

    mul_penalty[0] = 0.96;
    mul_penalty[1] = 1.0;
    mul_penalty[2] = 0.96;

    hanming_window = Hanming_weight(272, 272) * 0.176;

    m_zFeat = zeros({1 ,256 ,6 , 6}, torch::kFloat32).to(at::kCUDA);
    m_xFeat = zeros({3 ,256 ,22 , 22}, torch::kFloat32).to(at::kCUDA);
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
}


void Tracker::Init(const Mat& img, const Rect2d& roi)
{
    // Mat img1, img2, img3, img4;
    // img.convertTo(img1, CV_32FC3);
    // img1.copyTo(img2);
    // img2.setTo(0);
    // img2.convertTo(img4, CV_8UC3);
    // imshow("before", img4);
    // at::Tensor tmplate;
    // tmplate = torch::from_blob(img1.data, { 1, img1.rows, img1.cols, img.channels() }, torch::kFloat32).permute({ 0, 2, 1, 3 }).to(at::kCUDA);
    // int bs = img1.rows * img1.cols * img.channels() * sizeof(float);
    // cudaStream_t stream1;
    // cudaStreamCreate(&stream1);
    // cudaMemcpyAsync(img2.data, tmplate.data_ptr(), bs, cudaMemcpyDeviceToHost, stream1);
    // cudaStreamSynchronize(stream1);
    // img2.convertTo(img3, CV_8UC3);
    // imshow("after", img3);
    // waitKey();



    // exemplar and search sizes
    float context = (roi.width + roi.height) * .5f;
    m_zSize = sqrt((roi.width + context) * (roi.height + context));
    m_xSize = m_zSize * 2.0f;

    // prepare input data
    Mat z;
    PreProcess(img, z, roi, m_zSize, 127);

    // vector<Mat> sps;
    // sps.resize(3);
    // split(z, sps);
    // vconcat(sps, z);
    // imshow("cpp", z);
    // waitKey();


    z.convertTo(z, CV_32FC3);

    Tensor tz = torch::from_blob(z.data, {1, 127, 127, 3}, torch::kFloat32).to(at::kCUDA).permute({0, 3, 1, 2}).contiguous();
    // Tensor tz1 = tz.permute({0, 3, 1, 2});
    cout << tz.is_contiguous() << endl;
    // cout << tz1.is_contiguous() << endl;
    // tz1 = tz1.contiguous();
    // cout << tz1.is_contiguous() << endl;
    
    // allocate buffers
    Dims inputDims = mEngine->getProfileDimensions(0, 0, OptProfileSelector::kMIN);
    mContext->setBindingDimensions(0, inputDims);
    Dims outputDims = mContext->getBindingDimensions(1);

 
    int inputSize = inputDims.d[0] * inputDims.d[1] * inputDims.d[2] * inputDims.d[3];
    int inputByteSize = inputSize * sizeof(float);
    cudaMalloc(&m_inputDeviceBuffer, inputByteSize);
    m_inputHostBuffer = malloc(inputByteSize);
    outputSize = outputDims.d[0] * outputDims.d[1] * outputDims.d[2] * outputDims.d[3];
    outputByteSize = outputSize * sizeof(float);
    cudaMalloc(&m_outputDeviceBuffer, outputByteSize);
    m_outputHostBuffer = malloc(outputByteSize);
    mDeviceBindings.clear();
    // mDeviceBindings.emplace_back(m_inputDeviceBuffer);
    // mDeviceBindings.emplace_back(m_outputDeviceBuffer);
    mDeviceBindings.emplace_back(tz.data_ptr());
    mDeviceBindings.emplace_back(m_zFeat.data_ptr());

    // Asynchronously copy data from host input buffers to device input buffers
    // cudaMemcpyAsync(m_inputDeviceBuffer, m_inputHostBuffer, inputByteSize, cudaMemcpyHostToDevice, m_stream);
    // cudaMemcpyAsync(m_inputDeviceBuffer, z.data, inputByteSize, cudaMemcpyHostToDevice, m_stream);

    // Asynchronously enqueue the inference work
    //mContext->enqueue(1, mDeviceBindings.data(), m_stream, nullptr);
    mContext->enqueueV2(mDeviceBindings.data(), m_stream, nullptr);

    // Asynchronously copy data from device output buffers to host output buffers
    // cudaMemcpyAsync(m_outputHostBuffer, m_outputDeviceBuffer, outputByteSize, cudaMemcpyDeviceToHost, m_stream);
    // cudaMemcpyAsync(m_outputHostBuffer, m_zFeat.data_ptr(), outputByteSize, cudaMemcpyDeviceToHost, m_stream);

    // Wait for the work in the m_stream to complete
    cudaStreamSynchronize(m_stream);

    cout << m_zFeat.max().to(kCPU) << endl;
    cout << m_zFeat.min().to(kCPU) << endl;

    // float minv = 100;
    // float maxv = -100;
    // float* prob = (float*)m_outputHostBuffer;
    // for (int i = 0; i < outputSize; i++)
    // {
    //     if(minv > prob[i]){
    //         minv = prob[i];
    //     }
    //     if(maxv < prob[i]){
    //         maxv = prob[i];
    //     }
    // }
    // cout << maxv << endl;
    // cout << minv << endl;


    inputDims = mEngine->getProfileDimensions(0, 0, OptProfileSelector::kMAX);
    mContext->setBindingDimensions(0, inputDims);
    outputDims = mContext->getBindingDimensions(1);

    mDeviceBindings[1] = m_xFeat.data_ptr();
    outputSize = outputDims.d[0] * outputDims.d[1] * outputDims.d[2] * outputDims.d[3];
    outputByteSize = outputSize * sizeof(float);
    free(m_outputHostBuffer);
    m_outputHostBuffer = malloc(outputByteSize);

    // width = roi.width;
    // height = roi.height;
    // center_x = roi.x + int(width / 2);
    // center_y = roi.y + int(height / 2);

    // // �������������С
    // float context = (width + height) * .5;
    // target_sz = pow((width + context) * (height + context), 0.5);
    // search_sz = target_sz * 2.0;

    // // ��ȡĿ��ģ��ͼ��
    // Mat tpla;
    // Rect2d init_roi(center_x - int(target_sz / 2), center_y - int(target_sz / 2), target_sz, target_sz);
    // tpla = img(init_roi).clone();

    // // ���ŵ���׼ģ��ͼ���С����ת����������
    // tpla.convertTo(tpla, CV_32FC3);
    // resize(tpla, tpla, cv::Size(127, 127));
    // int kCHANNELS = tpla.channels();
    // int frame_h = tpla.rows;
    // int frame_w = tpla.cols;

    // // ģ�����ݷŽ��Դ�
    // tmplate = torch::from_blob(tpla.data, { 1, frame_h, frame_w, kCHANNELS }, torch::kFloat32);
    // tmplate = tmplate.permute({ 0, 3, 1, 2 }).to(at::kCUDA);

    // /*
    // std::vector<torch::jit::IValue> data;
    // data.push_back(tmplate);
    // data.push_back(torch::tensor(1.0));
    // My_module.forward(data);
    // */
}


void Tracker::Update(const Mat& img, Rect2d& roi)
{   
    vector<Mat> xs;
    xs.resize(3);
    for (int i = 0; i < 3; i++) {
        PreProcess(img, xs[i], roi, m_zSize * m_scales[i], 255);
    }
    Mat x;
    vconcat(xs, x);
    x.convertTo(x, CV_32FC3);
    Tensor tx = torch::from_blob(x.data, {3, 255, 255, 3}, torch::kFloat32).permute({0, 3, 1, 2}).to(at::kCUDA);
    mDeviceBindings[0] = tx.data_ptr();

    // Asynchronously enqueue the inference work
    mContext->enqueue(1, mDeviceBindings.data(), m_stream, nullptr);

    // Asynchronously copy data from device output buffers to host output buffers
    //cudaMemcpyAsync(m_outputHostBuffer, m_outputDeviceBuffer, outputByteSize, cudaMemcpyDeviceToHost, m_stream);
    cudaMemcpyAsync(m_outputHostBuffer, m_xFeat.data_ptr(), outputByteSize, cudaMemcpyDeviceToHost, m_stream);

    // Wait for the work in the m_stream to complete
    cudaStreamSynchronize(m_stream);




    // float* prob = (float*)m_outputHostBuffer;
    // for (int i = 0; i < outputSize; i++)
    // {
    //     cout << i << "\t" << prob[i] << endl;
    // }



    // // ׼��ģ����������
    // vector<torch::Tensor> search_mul_tensor;
    // for (int i = 0; i < 3; i++)
    // {
    //     // ������������
    //     Rect2d search_roi;
    //     search_roi.x = center_x - int(search_sz * mul_scale[i] / 2.0);
    //     search_roi.y = center_y - int(search_sz * mul_scale[i] / 2.0);
    //     search_roi.width = int(search_sz * mul_scale[i]);
    //     search_roi.height = int(search_sz * mul_scale[i]);

    //     // ͼ�񲹱��Ա��ȡ����ͼ��
    //     int top = 0, bottom = 0, left = 0, right = 0;
    //     if (search_roi.x + search_roi.width > img.cols)
    //     {
    //         right = int(search_roi.x + search_roi.width - img.cols) + 1;
    //     }
    //     if (search_roi.x < 0)
    //     {
    //         left = int(-search_roi.x) + 1;
    //     }
    //     if (search_roi.y + search_roi.height > img.rows)
    //     {
    //         bottom = int(search_roi.y + search_roi.height - img.rows) + 1;
    //     }
    //     if (search_roi.y < 0)
    //     {
    //         top = int(-search_roi.y) + 1;
    //     }

    //     //cudaDeviceSynchronize();
    //     //TickMeter tm;
    //     //tm.start();
    //     Mat imgPadding;
    //     copyMakeBorder(img, imgPadding, top, bottom, left, right, cv::BORDER_REPLICATE);
    //     search_roi.x += left;
    //     search_roi.y += top;
    //     //cudaDeviceSynchronize();
    //     //tm.stop();
    //     //cout << "timecost: " << tm.getTimeMilli() << endl;

    //     // ��ȡ����ͼ��
    //     Mat imgSearch = imgPadding(search_roi).clone();

    //     // ���ŵ���׼����ͼ���С��ת����������
    //     imgSearch.convertTo(imgSearch, CV_32FC3);
    //     resize(imgSearch, imgSearch, cv::Size(255, 255));
    //     int kCHANNELS = imgSearch.channels();
    //     int frame_h = imgSearch.rows;
    //     int frame_w = imgSearch.cols;



    //     // �����������ݷŽ��Դ�
    //     torch::Tensor search = torch::from_blob(imgSearch.data, { 1, frame_h, frame_w, kCHANNELS }, torch::kFloat32);
    //     search = search.permute({ 0, 3, 1, 2 }).to(at::kCUDA);
    //     search_mul_tensor.push_back(search);

    // }

    // torch::TensorList TL = torch::TensorList(search_mul_tensor);
    // torch::Tensor mul_tensor = torch::cat(TL, 0);

    // // ׼����������
    // std::vector<torch::jit::IValue> data;
    // data.push_back(tmplate);
    // data.push_back(mul_tensor);

    // // Ԥ����Ӧͼ
    // //cudaDeviceSynchronize();
    // //TickMeter tm;
    // //tm.start();
    // at::Tensor output = module.forward(data).toTensor();
    // //cudaDeviceSynchronize();
    // //tm.stop();
    // //cout << "timecost: " << tm.getTimeMilli() << endl;

    // // ��Ӧͼ���ݷ����ڴ�
    // cv::Mat resultImg(cv::Size(17, 17), CV_32FC3);
    // output = output.squeeze().detach().permute({ 1, 2, 0 }).to(torch::kCPU).to(torch::kFloat32);
    // memcpy((void*)resultImg.data, output.data_ptr(), 4 * output.numel());
    // //Mat resultImg(cv::Size(17,17), CV_8UC3, output.data<float>());

    // // �ָ��߶�
    // resize(resultImg, resultImg, cv::Size(272, 272), cv::INTER_CUBIC);

    // vector<Mat> channels;
    // split(resultImg, channels);
    // double global_max = -100000.0;
    // double global_min = 100000.0;
    // int gloabl_maxInd[3];
    // int choosed_id = 1;
    // for (int i = 0; i < channels.size(); i++)
    // {
    //     int maxInd[3];
    //     double max_value;
    //     double min_value;
    //     minMaxIdx(channels[i] * mul_penalty[i] * mul_penalty[i], &min_value, &max_value, NULL, maxInd);
    //     // cout<<max_value<<" M M M  "<<endl;
    //     if (max_value > global_max)
    //     {
    //         global_max = max_value;
    //         gloabl_maxInd[0] = maxInd[0];
    //         gloabl_maxInd[1] = maxInd[1];
    //         gloabl_maxInd[2] = i;
    //     }
    //     if (min_value < global_min)
    //     {
    //         global_min = min_value;
    //         choosed_id = i;
    //     }
    // }
    // resultImg = channels[choosed_id];
    // resultImg -= global_min;
    // resultImg /= sum(resultImg)[0];

    // resultImg = resultImg * 0.824 + hanming_window;

    // split(resultImg, channels);
    // for (int i = 0; i < channels.size(); i++)
    // {
    //     int maxInd[3];
    //     double max_value;
    //     double min_value;
    //     minMaxIdx(channels[i], &min_value, &max_value, NULL, maxInd);
    //     if (max_value > global_max)
    //     {
    //         global_max = max_value;
    //         gloabl_maxInd[0] = maxInd[0];
    //         gloabl_maxInd[1] = maxInd[1];
    //     }
    //     if (min_value < global_min)
    //     {
    //         global_min = min_value;
    //     }
    // }

    // float dispx = gloabl_maxInd[1] - 271 / 2.0;
    // float dispy = gloabl_maxInd[0] - 271 / 2.0;
    // dispx /= 2.0;
    // dispy /= 2.0;
    // dispx = dispx * search_sz / 255;
    // dispy = dispy * search_sz / 255;


    // search_sz *= mul_scale[gloabl_maxInd[2]];
    // center_x += dispx;
    // center_y += dispy;

    // width *= mul_scale[gloabl_maxInd[2]];
    // height *= mul_scale[gloabl_maxInd[2]];

    // // ����Ŀ���
    // roi.x = center_x - int(width / 2);
    // roi.y = center_y - int(height / 2);
    // roi.width = width;
    // roi.height = height;
}


Mat Hanming_weight(int height, int width)
{

    cv::Mat hanming_weight(cv::Size(height, width), CV_32FC1);

    cv::Mat hanming_vector_row(cv::Size(height, 1), CV_32FC1);
    cv::Mat hanming_vector_col(cv::Size(1, width), CV_32FC1);
    for (int i = 0; i < height; i++)
    {
        hanming_vector_row.at<float>(0, i) = pow(sin(CV_PI * (float(i) / height)), 2);
    }
    for (int i = 0; i < width; i++)
    {
        hanming_vector_col.at<float>(i, 0) = pow(sin(CV_PI * (float(i) / width)), 2);
    }

    float sum = 0.0;
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            hanming_weight.at<float>(i, j) = hanming_vector_row.at<float>(0, i) * hanming_vector_col.at<float>(j, 0);
            sum += hanming_weight.at<float>(i, j);
        }
    }

    hanming_weight = hanming_weight / sum;

    return hanming_weight;
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
    resize(dst, dst, Size(outSize, outSize));
}