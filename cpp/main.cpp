#include <vector>
#include <set>
#include <fstream>
#include <iostream>
#include <assert.h>
#include <NvInfer.h>

using namespace std;
using namespace nvinfer1;


class Logger : public ILogger
 {
     void log(Severity severity, const char* msg) override
     {
         // suppress info-level messages
         if (severity != Severity::kINFO)
             cout << msg << endl;
     }
 } gLogger;


int main()
{
    std::vector<char> trtModelStream_;
    size_t size{ 0 };

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
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream_.data(), size, nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    
    // int num_model_bindings = engine->getNbBindings();
    // int num_profiles = engine->getNbOptimizationProfiles();
    // std::set<int> supported_profiles;
    // for (int profile = 0; profile < num_profiles; profile++) {
    //   bool supports_batching = true;
    //   for (int binding = 0; binding < num_model_bindings; binding++) {
    //     int effective_binding_index = binding + (profile * num_model_bindings);
    //     if (engine->bindingIsInput(effective_binding_index)) {
    //       if (!engine->isShapeBinding(effective_binding_index)) {
    //         nvinfer1::Dims min_shape = engine->getProfileDimensions(
    //             effective_binding_index, profile,
    //             nvinfer1::OptProfileSelector::kMIN);
    //         if (min_shape.d[0] != 1) {
    //           supports_batching = false;
    //           break;
    //         }
    //       } else {
    //         const int32_t* shapes = engine->getProfileShapeValues(
    //             effective_binding_index, profile,
    //             nvinfer1::OptProfileSelector::kMIN);
    //         if (*shapes != 1) {
    //           supports_batching = false;
    //           break;
    //         }
    //       }
    //     }
    //   }
    //   if (supports_batching) {
    //     supported_profiles.insert(profile);
    //   }
    // }

    Dims min_shape = engine->getProfileDimensions(0, 0, OptProfileSelector::kMIN);
    Dims max_shape = engine->getProfileDimensions(0, 0, OptProfileSelector::kMAX);


    return 0;
}