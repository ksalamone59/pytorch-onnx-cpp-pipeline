#ifndef ORT_INFERENCE_FROM_MODEL_H
#define ORT_INFERENCE_FROM_MODEL_H

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <memory>
#include <numeric>
#include <random>
#include "onnxruntime_cxx_api.h"

// Define both of these globally and static so only one copy exists 
// And they are easily accessible 
inline static Ort::Env& getONNXEnv() noexcept
{
    static Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "ONNXModel");
    return env;
}
inline static Ort::SessionOptions& getSessionOptions() noexcept 
{
    static Ort::SessionOptions session_options = [](){  
        Ort::SessionOptions options;  
        options.SetIntraOpNumThreads(1);
        options.SetInterOpNumThreads(1);
        options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        options.EnableMemPattern(); 
        options.EnableCpuMemArena();
        return options;
    }();
    return session_options;
}

class ONNXModel
{
    private:
        std::unique_ptr<Ort::Session> session;
        Ort::ShapeInferContext::Ints input_shape;
        std::vector<std::string> input_strs, output_strs;
        std::vector<const char *> input_names, output_names;
        std::size_t input_count{0}, output_count{0}, input_size{1};
        Ort::Value input_tensor;
        std::vector<float> input_data;
        float *input_ptr{nullptr};
        Ort::RunOptions run_options{nullptr};
    public:
        ONNXModel() noexcept = default;
        ~ONNXModel() noexcept = default;
        ONNXModel(ONNXModel&&) noexcept = default;
        ONNXModel& operator=(ONNXModel&&) noexcept = default;
        ONNXModel(const std::string &modelPath) 
        {
            session = std::make_unique<Ort::Session>(getONNXEnv(), modelPath.c_str(), getSessionOptions());
            input_count = session->GetInputCount();
            output_count = session->GetOutputCount();
            input_shape = session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
            for(auto &dim : input_shape)
            {
                if(dim < 0) dim = 1; 
            }
            input_size = std::accumulate(input_shape.begin(), input_shape.end(), 1LL, std::multiplies<std::size_t>());
            input_strs = session->GetInputNames();
            output_strs = session->GetOutputNames();
            input_names.reserve(input_count);
            output_names.reserve(output_count);
            for (std::size_t i = 0; i < input_count; i++)
            {
                input_names.push_back(input_strs[i].c_str());
            }
            for(std::size_t i = 0; i < output_count; i++)
            {
                output_names.push_back(output_strs[i].c_str());
            }
            Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeCPU);
            input_data.resize(input_size);
            input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_data.data(), input_data.size(), input_shape.data(), input_shape.size());
            input_ptr = input_tensor.GetTensorMutableData<float>();
            std::memset(input_ptr, 0, input_size * sizeof(float));        
        }
        inline void reset_input_tensor() noexcept 
        {
            std::memset(input_ptr, 0, input_size * sizeof(float));
        }
        inline void set_input_tensor(const std::vector<float> &data) noexcept 
        {
            std::memcpy(input_ptr, data.data(), input_size * sizeof(float));
        }
        inline std::vector<Ort::Value> run()
        {
            return session->Run(run_options, input_names.data(), &input_tensor, 1, output_names.data(), output_count);
        }
    inline std::vector<Ort::Value> run_batch(const std::vector<float> &batch_data, std::size_t batch_size)
    {
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeCPU);
        std::vector<int64_t> batch_shape = {static_cast<int64_t>(batch_size), 1};
        Ort::Value batch_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            const_cast<float*>(batch_data.data()),
            batch_data.size(),
            batch_shape.data(),
            batch_shape.size()
        );
        return session->Run(run_options, input_names.data(), &batch_tensor, 1, output_names.data(), output_count);
    }
};

class coordinateData
{
    private:
        std::vector<float> x, y;
    public:
        coordinateData() noexcept = default;
        ~coordinateData() noexcept = default;
        coordinateData(coordinateData&&) noexcept = default;
        coordinateData& operator=(coordinateData&&) noexcept = default;
        coordinateData(const std::vector<float> &x, const std::vector<float> &y) : x(x), y(y) {}
        inline const std::vector<float>& get_x() const noexcept {return x;}
        inline const std::vector<float>& get_y() const noexcept {return y;}
};

inline float function(float x) noexcept 
{
    return std::sin(x) * std::exp(-0.1f * x * x);
}
coordinateData generate_model_data(std::size_t num_points, float noise_stddev = 1.f, float x0 = -10.f, float xf = 10.f);
void exportData(const coordinateData &data, const std::string &filename);

#endif 