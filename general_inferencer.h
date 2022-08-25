

#ifndef GENERAL_INFERENCER_H
#define GENERAL_INFERENCER_H
#include "tflite_inference.h"

struct TensorToPassOn
{
    std::string name;
    std::vector<int32_t> dimensions;
    float *flatData;
};

class generalInferencer : public tflite_inference_t
{
    generalInferencer();
    virtual ~generalInferencer();

    int init(
        const std::string &modelPath,
        std::vector<std::string> outLayerNames,
        std::vector<int> imageDimensions;
        int use_nnapi = 2,
        int num_threads = 4);

    void inference(const std::vector<uint8_t> &inputImage, std::vector<TensorToPassOn> &outResults);

private:
    std::vector<std::string> mOutLayerNames; // Names of the network to be output

    std::vector<int> mInputImageDimension;
}

#endif