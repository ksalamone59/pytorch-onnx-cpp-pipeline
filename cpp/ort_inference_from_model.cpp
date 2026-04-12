#include "ort_inference_from_model.h"

void exportData(const coordinateData &data, const std::string &filename)
{
    std::ofstream file(filename);
    if (!file.is_open())
    {
        throw std::runtime_error("Could not open file for writing: " + filename);
    }
    const auto &x = data.get_x();
    const auto &y = data.get_y();
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        file << x[i] << " " << y[i] << "\n";
    }
    file.close();
}

coordinateData generate_model_data(std::size_t num_points, float noise_stddev, float x0, float xf)
{
    // Could also just resize the vectors here and use like arrays
    // Choosing reserve for more realistic usage where num_points might not be known 
    // In a different situation
    std::vector<float> x, y;
    x.reserve(num_points);
    y.reserve(num_points);
    std::mt19937 rng(std::random_device{}());
    std::normal_distribution<float> noise(0.0f, noise_stddev);
    float step_size_x = (xf - x0) / static_cast<float>(num_points - 1);
    for (std::size_t i = 0; i < num_points; ++i)
    {
        float x_val = x0 + i * step_size_x;
        float y_val = function(x_val);
        // Apply heteroscedastic noise
        y_val += noise(rng) * 0.1f * std::fabs(y_val);
        x.push_back(x_val);
        y.push_back(y_val); // Use the function and add noise
    }
    return coordinateData(std::move(x), std::move(y));
}

int main()
{
    ONNXModel model("../../model_files/function_model.onnx");
    auto data = generate_model_data(1000);
    exportData(data, "../../Plotting/ort_inference/cpp_input_data.dat");

    auto providers = Ort::GetAvailableProviders();
    // My version of ORT C++ doesn't have GPU support because of my experimental requirements 
    // But this is how you'd check and use a GPU
    bool useGPU = std::find(providers.begin(), providers.end(), "CUDAExecutionProvider") != providers.end();
    for(const auto &provider : providers)
    {
        std::cout << "Available provider: " << provider << std::endl;
    }
    std::cout << "Use GPU? " << useGPU << std::endl;

    // Predict at each x

    // Batched inference 
    // auto x_data = data.get_x();
    // auto output = model.run_batch(x_data, x_data.size());
    // auto output_data = output.front().GetTensorData<float>();
    // std::vector<float> y_vals(output_data, output_data + x_data.size());
    
    // Per-point inference
    std::vector<float> y_vals;
    y_vals.reserve(data.get_x().size());
    for(const auto x_val : data.get_x())
    {
        model.set_input_tensor({x_val});
        auto output = model.run();
        auto &output_tensor = output.front();
        auto output_data = output_tensor.GetTensorData<float>();
        y_vals.push_back(output_data[0]);
    }
    exportData(coordinateData(data.get_x(), y_vals), "../../Plotting/ort_inference/ort_inference.dat");
    auto realFunctionData = generate_model_data(1000, 0.f);
    exportData(realFunctionData, "../../Plotting/ort_inference/real_function.dat");
    return 0;
}