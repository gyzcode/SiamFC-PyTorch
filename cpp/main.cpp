//#include <vector>
//#include <set>
//#include <numeric>
//#include <assert.h>
//#include <NvInfer.h>
//#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>


#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"

#include "NvCaffeParser.h"
#include "NvInfer.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <sstream>


using namespace std;
using namespace nvinfer1;


const std::string gSampleName = "TensorRT.sample_mnist";

//!
//! \brief  The SampleMNIST class implements the MNIST sample
//!
//! \details It creates the network using a trained Caffe MNIST classification model
//!
class SampleMNIST
{
    template <typename T>
    using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
    SampleMNIST(const samplesCommon::CaffeSampleParams& params)
        : mParams(params)
    {
    }

    //!
    //! \brief Builds the network engine
    //!
    bool build();

    //!
    //! \brief Runs the TensorRT inference engine for this sample
    //!
    bool infer();

    //!
    //! \brief Used to clean up any state created in the sample class
    //!
    bool teardown();

private:
    //!
    //! \brief uses a Caffe parser to create the MNIST Network and marks the
    //!        output layers
    //!
    bool constructNetwork(
        SampleUniquePtr<nvcaffeparser1::ICaffeParser>& parser, SampleUniquePtr<nvinfer1::INetworkDefinition>& network);

    //!
    //! \brief Reads the input and mean data, preprocesses, and stores the result in a managed buffer
    //!
    bool processInput(
        const samplesCommon::BufferManager& buffers, const std::string& inputTensorName, int inputFileIdx) const;

    bool processInput1(
        const samplesCommon::BufferManager& buffers, const std::string& inputTensorName, const std::string& inputFileName) const;

    //!
    //! \brief Verifies that the output is correct and prints it
    //!
    bool verifyOutput(
        const samplesCommon::BufferManager& buffers, const std::string& outputTensorName, int groundTruthDigit) const;

    bool verifyOutput1(
        const samplesCommon::BufferManager& buffers, const std::string& outputTensorName) const;        

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine{nullptr}; //!< The TensorRT engine used to run the network

    samplesCommon::CaffeSampleParams mParams; //!< The parameters for the sample.

    nvinfer1::Dims mInputDims; //!< The dimensions of the input to the network.

    SampleUniquePtr<nvcaffeparser1::IBinaryProtoBlob>
        mMeanBlob; //! the mean blob, which we need to keep around until build is done
};

//!
//! \brief Creates the network, configures the builder and creates the network engine
//!
//! \details This function creates the MNIST network by parsing the caffe model and builds
//!          the engine that will be used to run MNIST (mEngine)
//!
//! \return Returns true if the engine was created successfully and false otherwise
//!
bool SampleMNIST::build()
{
    size_t size{ 0 };
    std::vector<char> trtModelStream_;
    std::ifstream file("/home/gyz/workzone/siamfc-pytorch/pretrained/siamfc_alexnet_e50_dynamic.engine", std::ios::binary);
    if (file.good())
    {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream_.resize(size);
        std::cout << "size" << trtModelStream_.size() << std::endl;
        file.read(trtModelStream_.data(), size);
        file.close();
    }
    std::cout << "size" << size;
    IRuntime* runtime = createInferRuntime(sample::gLogger.getTRTLogger());
    assert(runtime != nullptr);
    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
    runtime->deserializeCudaEngine(trtModelStream_.data(), size, nullptr), samplesCommon::InferDeleter());

    if (!mEngine)
        return false;
    else
        return true;
}


//!
//! \brief Reads the input and mean data, preprocesses, and stores the result in a managed buffer
//!
bool SampleMNIST::processInput(
    const samplesCommon::BufferManager& buffers, const std::string& inputTensorName, int inputFileIdx) const
{
    const int inputH = mInputDims.d[1];
    const int inputW = mInputDims.d[2];

    // Read a random digit file
    srand(unsigned(time(nullptr)));
    std::vector<uint8_t> fileData(inputH * inputW);
    readPGMFile(locateFile(std::to_string(inputFileIdx) + ".pgm", mParams.dataDirs), fileData.data(), inputH, inputW);

    // Print ASCII representation of digit
    sample::gLogInfo << "Input:\n";
    for (int i = 0; i < inputH * inputW; i++)
    {
        sample::gLogInfo << (" .:-=+*#%@"[fileData[i] / 26]) << (((i + 1) % inputW) ? "" : "\n");
    }
    sample::gLogInfo << std::endl;

    float* hostInputBuffer = static_cast<float*>(buffers.getHostBuffer(inputTensorName));

    for (int i = 0; i < inputH * inputW; i++)
    {
        hostInputBuffer[i] = float(fileData[i]);
    }

    return true;
}


bool SampleMNIST::processInput1(
    const samplesCommon::BufferManager& buffers, const std::string& inputTensorName, const std::string& inputFileName) const
{
    const int inputN = mInputDims.d[0];
    const int inputC = mInputDims.d[1];
    const int inputH = mInputDims.d[2];
    const int inputW = mInputDims.d[3];

    // Read a random digit file
    srand(unsigned(time(nullptr)));
    int size = inputN * inputC * inputH * inputW;
    int byteSize = size * sizeof(float);
    cv::Mat frame = cv::imread(inputFileName);
    cv::resize(frame, frame, cv::Size(255, 255));
    vector<cv::Mat> c;
    c.resize(3);
    cv::split(frame, c);
    cv::Mat cc;
    cv::vconcat(c, cc);
    cv::Mat cc1;
    cv::vconcat(cc, cc, cc1);
    cv::vconcat(cc, cc1, cc1);
    cv::imshow("test", cc1);
    cv::waitKey();
    cv::Mat cc2;
    cc1.convertTo(cc2, CV_32F);
    float* fileData = (float*)cc2.data;

    float* hostInputBuffer = static_cast<float*>(buffers.getHostBuffer(inputTensorName));

    // for (int i = 0; i < size; i++)
    // {
    //     hostInputBuffer[i] = fileData[i];
    // }
    memcpy(hostInputBuffer, fileData, byteSize);

    return true;
}


//!
//! \brief Verifies that the output is correct and prints it
//!
bool SampleMNIST::verifyOutput(
    const samplesCommon::BufferManager& buffers, const std::string& outputTensorName, int groundTruthDigit) const
{
    const float* prob = static_cast<const float*>(buffers.getHostBuffer(outputTensorName));

    // Print histogram of the output distribution
    sample::gLogInfo << "Output:\n";
    float val{0.0f};
    int idx{0};
    const int kDIGITS = 10;

    for (int i = 0; i < kDIGITS; i++)
    {
        if (val < prob[i])
        {
            val = prob[i];
            idx = i;
        }

        sample::gLogInfo << i << ": " << std::string(int(std::floor(prob[i] * 10 + 0.5f)), '*') << "\n";
    }
    sample::gLogInfo << std::endl;

    return (idx == groundTruthDigit && val > 0.9f);
}

bool SampleMNIST::verifyOutput1(
    const samplesCommon::BufferManager& buffers, const std::string& outputTensorName) const
{
    const float* prob = static_cast<const float*>(buffers.getHostBuffer(outputTensorName));

    // Print histogram of the output distribution
    sample::gLogInfo << "Output:\n";
    float val{0.0f};
    int idx{0};
    const int size = buffers.size(outputTensorName) / sizeof(float);

    for (int i = 0; i < size; i++)
    {
        if(!isnan(prob[i]))
            cout << i << "\t" << prob[i] << endl;
    }
    sample::gLogInfo << std::endl;

    return true;
}

//!
//! \brief Uses a caffe parser to create the MNIST Network and marks the
//!        output layers
//!
//! \param network Pointer to the network that will be populated with the MNIST network
//!
//! \param builder Pointer to the engine builder
//!
bool SampleMNIST::constructNetwork(
    SampleUniquePtr<nvcaffeparser1::ICaffeParser>& parser, SampleUniquePtr<nvinfer1::INetworkDefinition>& network)
{
    const nvcaffeparser1::IBlobNameToTensor* blobNameToTensor = parser->parse(
        mParams.prototxtFileName.c_str(), mParams.weightsFileName.c_str(), *network, nvinfer1::DataType::kFLOAT);

    for (auto& s : mParams.outputTensorNames)
    {
        network->markOutput(*blobNameToTensor->find(s.c_str()));
    }

    // add mean subtraction to the beginning of the network
    nvinfer1::Dims inputDims = network->getInput(0)->getDimensions();
    mMeanBlob
        = SampleUniquePtr<nvcaffeparser1::IBinaryProtoBlob>(parser->parseBinaryProto(mParams.meanFileName.c_str()));
    nvinfer1::Weights meanWeights{nvinfer1::DataType::kFLOAT, mMeanBlob->getData(), inputDims.d[1] * inputDims.d[2]};
    // For this sample, a large range based on the mean data is chosen and applied to the head of the network.
    // After the mean subtraction occurs, the range is expected to be between -127 and 127, so the rest of the network
    // is given a generic range.
    // The preferred method is use scales computed based on a representative data set
    // and apply each one individually based on the tensor. The range here is large enough for the
    // network, but is chosen for example purposes only.
    float maxMean
        = samplesCommon::getMaxValue(static_cast<const float*>(meanWeights.values), samplesCommon::volume(inputDims));

    auto mean = network->addConstant(nvinfer1::Dims3(1, inputDims.d[1], inputDims.d[2]), meanWeights);
    if (!mean->getOutput(0)->setDynamicRange(-maxMean, maxMean))
    {
        return false;
    }
    if (!network->getInput(0)->setDynamicRange(-maxMean, maxMean))
    {
        return false;
    }
    auto meanSub = network->addElementWise(*network->getInput(0), *mean->getOutput(0), ElementWiseOperation::kSUB);
    if (!meanSub->getOutput(0)->setDynamicRange(-maxMean, maxMean))
    {
        return false;
    }
    network->getLayer(0)->setInput(0, *meanSub->getOutput(0));
    samplesCommon::setAllTensorScales(network.get(), 127.0f, 127.0f);

    return true;
}

//!
//! \brief Runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample. It allocates
//!          the buffer, sets inputs, executes the engine, and verifies the output.
//!
bool SampleMNIST::infer()
{
    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }
    context->setBindingDimensions(0, mEngine->getProfileDimensions(0, 0, OptProfileSelector::kMIN));
    mInputDims = context->getBindingDimensions(0);

    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(mEngine, mParams.batchSize, context.get());


    // Pick a random digit to try to infer
    srand(time(NULL));

    if (!processInput1(buffers, mParams.inputTensorNames[0], "/home/gyz/Pictures/person.jpg"))
    {
        return false;
    }
    // Create CUDA stream for the execution of this inference.
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // Asynchronously copy data from host input buffers to device input buffers
    buffers.copyInputToDeviceAsync(stream);

    // Asynchronously enqueue the inference work
    if (!context->enqueue(mParams.batchSize, buffers.getDeviceBindings().data(), stream, nullptr))
    {
        return false;
    }
    // Asynchronously copy data from device output buffers to host output buffers
    buffers.copyOutputToHostAsync(stream);

    // Wait for the work in the stream to complete
    cudaStreamSynchronize(stream);

    // Release stream
    cudaStreamDestroy(stream);

    // Check and print the output of the inference
    // There should be just one output tensor
    assert(mParams.outputTensorNames.size() == 1);
    bool outputCorrect = verifyOutput1(buffers, mParams.outputTensorNames[0]);

    //return outputCorrect;
    return true;
}

//!
//! \brief Used to clean up any state created in the sample class
//!
bool SampleMNIST::teardown()
{
    //! Clean up the libprotobuf files as the parsing is complete
    //! \note It is not safe to use any other part of the protocol buffers library after
    //! ShutdownProtobufLibrary() has been called.
    nvcaffeparser1::shutdownProtobufLibrary();
    return true;
}

//!
//! \brief Initializes members of the params struct using the command line args
//!
samplesCommon::CaffeSampleParams initializeSampleParams(const samplesCommon::Args& args)
{
    samplesCommon::CaffeSampleParams params;
    if (args.dataDirs.empty()) //!< Use default directories if user hasn't provided directory paths
    {
        params.dataDirs.push_back("data/mnist/");
        params.dataDirs.push_back("data/samples/mnist/");
    }
    else //!< Use the data directory provided by the user
    {
        params.dataDirs = args.dataDirs;
    }

    params.prototxtFileName = locateFile("mnist.prototxt", params.dataDirs);
    params.weightsFileName = locateFile("mnist.caffemodel", params.dataDirs);
    params.meanFileName = locateFile("mnist_mean.binaryproto", params.dataDirs);
    params.inputTensorNames.push_back("data");
    params.batchSize = 1;
    params.outputTensorNames.push_back("prob");
    params.dlaCore = args.useDLACore;
    params.int8 = args.runInInt8;
    params.fp16 = args.runInFp16;

    return params;
}


int main(int argc, char** argv)
{
    auto sampleTest = sample::gLogger.defineTest(gSampleName, argc, argv);

    samplesCommon::CaffeSampleParams params;
    params.inputTensorNames.push_back("input");
    params.outputTensorNames.push_back("output");
    SampleMNIST sample(params);
    sample::gLogInfo << "Building and running a GPU inference engine for MNIST" << std::endl;

    if (!sample.build())
    {
        return sample::gLogger.reportFail(sampleTest);
    }

    if (!sample.infer())
    {
        return sample::gLogger.reportFail(sampleTest);
    }

 
    
    return 0;
}