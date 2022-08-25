

#include "general_inferencer.h"

generalInferencer::generalInferencer()
{
}

generalInferencer::~generalInferencer()
{
    tflite_inference_t::init(model, use_nnapi, num_threads);
}

int generalInferencer::init(
    const std::string &modelPath,
    std::vector<std::string> outLayerNames,
    std::vector<int> imageDimensions;
    int use_nnapi,
    int num_threads)
{
    mInputImageDimension = imageDimensions;
    mOutLayerNames = outLayerNames;
    return tflite_inference_t::init(modelPath, use_nnapi, num_threads);
}

void generalInferencer::inference(const std::vector<uint8_t> &inputImage, std::vector<TensorToPassOn> &outResults)
{
    inference->setup_input_tensor(mInputImageDimension.at(0), mInputImageDimension.at(1), mInputImageDimension.at(2), (uint8_t *)inputImage.data());
    inference->inference();

    inference->
}