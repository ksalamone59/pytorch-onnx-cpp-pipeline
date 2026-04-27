// Same header b/c shared classes and functions
#include "ort_inference_from_model.h" 

int main()
{
    ONNXModel model("../../model_files/function_model.onnx");
    auto data = generate_model_data(1000);
    benchmarker bench(model, data.get_x());

    // Benchmark per-point inference
    bench.warmup();
    bench.time_per_point();
    bench.report("Per-point Inference");

    // Manually reset benchmarker state before batch inference
    bench.reset();

    // Benchmark batch inference
    bench.warmup_batch();
    bench.time_batch();
    bench.report("Batch Inference");

    return 0;
}