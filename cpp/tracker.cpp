#include "tracker.h"
#include <iostream>
#include <fstream>
//#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

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
    target_sz = 0.0;
    search_sz = 0.0;

    mul_scale[0] = 0.964;
    mul_scale[1] = 1.0;
    mul_scale[2] = 1.0375;

    mul_penalty[0] = 0.96;
    mul_penalty[1] = 1.0;
    mul_penalty[2] = 0.96;

    hanming_window = Hanming_weight(272, 272) * 0.176;
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


    Mat frame;
    resize(img, frame, Size(127, 127));
    frame.convertTo(frame, CV_32FC3);
    at::Tensor tmplate;
    tmplate = torch::from_blob(frame.data, { 1, frame.rows, frame.cols, frame.channels() }, torch::kFloat32).permute({0, 3, 1, 2}).to(at::kCUDA);


    // allocate buffers
    Dims inputDims = mEngine->getProfileDimensions(0, 0, OptProfileSelector::kMIN);
    mContext->setBindingDimensions(0, inputDims);
    Dims outputDims = mContext->getBindingDimensions(1);
    int inputSize = inputDims.d[0] * inputDims.d[1] * inputDims.d[2] * inputDims.d[3];
    int inputByteSize = inputSize * sizeof(float);
    cudaMalloc(&m_inputDeviceBuffer, inputByteSize);
    m_inputHostBuffer = malloc(inputByteSize);
    int outputSize = outputDims.d[0] * outputDims.d[1] * outputDims.d[2] * outputDims.d[3];
    int outputByteSize = outputSize * sizeof(float);
    cudaMalloc(&m_outputDeviceBuffer, outputByteSize);
    m_outputHostBuffer = malloc(outputByteSize);
    vector<void*> mDeviceBindings;
    //mDeviceBindings.emplace_back(m_inputDeviceBuffer);
    mDeviceBindings.emplace_back(tmplate.data_ptr());
    mDeviceBindings.emplace_back(m_outputDeviceBuffer);

    // prepare input data
    // Mat frame;
    // cv::resize(img, frame, cv::Size(127, 127));
    // vector<cv::Mat> c;
    // c.resize(3);
    // cv::split(frame, c);
    // cv::Mat cc;
    // cv::vconcat(c, cc);
    // // cv::Mat cc1;
    // // cv::vconcat(cc, cc, cc1);
    // // cv::vconcat(cc, cc1, cc1);
    // cv::imshow("test", cc);
    // cv::waitKey();
    // cv::Mat cc2;
    // cc.convertTo(cc2, CV_32F);
    // float* fileData = (float*)cc2.data;
    // memcpy(m_inputHostBuffer, fileData, inputByteSize);




    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Asynchronously copy data from host input buffers to device input buffers
    //cudaMemcpyAsync(m_inputDeviceBuffer, m_inputHostBuffer, inputByteSize, cudaMemcpyHostToDevice, stream);

    // Asynchronously enqueue the inference work
    mContext->enqueue(1, mDeviceBindings.data(), stream, nullptr);

    // Asynchronously copy data from device output buffers to host output buffers
    cudaMemcpyAsync(m_outputHostBuffer, m_outputDeviceBuffer, outputByteSize, cudaMemcpyDeviceToHost, stream);

    // Wait for the work in the stream to complete
    cudaStreamSynchronize(stream);

    float* prob = (float*)m_outputHostBuffer;
    for (int i = 0; i < outputSize; i++)
    {
        cout << i << "\t" << prob[i] << endl;
    }


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