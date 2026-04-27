#include "ort_inference_from_model.h"

double calculate_inference_r2(const std::vector<float> &x, const std::vector<float> &y)
{
    double ss_res = 0.0;
    double ss_tot = 0.0;
    double y_mean = 0.0;
    for(float xi : x)
    {
        y_mean += function(xi);
    }
    y_mean /= static_cast<double>(x.size());
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        double y_truth = function(x[i]);
        ss_res += (y_truth - y[i]) * (y_truth - y[i]);
        ss_tot += (y_truth - y_mean) * (y_truth - y_mean);
    }
    return 1.0 - (ss_res / ss_tot);
}

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
    auto x_data = data.get_x();
    auto output = model.run_batch(x_data, x_data.size());
    auto output_data = output.front().GetTensorData<float>();
    std::vector<float> y_vals(output_data, output_data + x_data.size());
    
    // Per-point inference
    // std::vector<float> y_vals;
    // y_vals.reserve(data.get_x().size());
    // for(const auto x_val : data.get_x())
    // {
    //     model.set_input_scalar(x_val);
    //     auto output = model.run();
    //     auto &output_tensor = output.front();
    //     auto output_data = output_tensor.GetTensorData<float>();
    //     y_vals.push_back(output_data[0]);
    // }
    exportData(coordinateData(data.get_x(), y_vals), "../../Plotting/ort_inference/ort_inference.dat");
    auto realFunctionData = generate_model_data(1000, 0.f);
    exportData(realFunctionData, "../../Plotting/ort_inference/real_function.dat");
    double r2 = calculate_inference_r2(data.get_x(), y_vals);
    std::cout << "Inference R^2: " << r2 << std::endl;
    return 0;
}