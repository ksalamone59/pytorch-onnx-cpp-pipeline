### Overview
A project that takes a simple model in PyTorch to learn $f(x) = \sin(x)e^{-0.1x^2}$, 
exports it to ONNX, and runs inference in C++ via ONNX Runtime - demonstrating how a 
trained model can be deployed in a performance-critical environment without a Python 
runtime dependency. Results are visualized using [gnuplot-latex-utils](https://github.com/ksalamone59/gnuplot_latex_utils), producing 
publication-ready plots directly from C++ output. Tested with ORT 1.22.1. 

![Inference Plot](final_inference.svg)

## Repository Layout 
```plaintext 
├── python/
├── cpp/
├── model_files/
├── python_figures/
├── Plotting/
├── Makefile
```

- `python`: Directory with generate_model.py; the script used to generate the data for, and train, the main model. It also exports the model to .onnx and .pth formats, and quantifies the performance of the model on various datasets.
- `cpp`: Directory where the C++ code for inference lies, ort_inference_from_model.cpp. Includes a CMakeLists.txt where you can compile the directory by creating a build directory then running `cmake -DONNXRUNTIME_DIR="/path/to/ONNXRunTime"`. Executable will be named onnx_inference. However, the Makefile will handle this should the user want less interfacing.
- `model_files`: Directory that stores output from generate_model.py, both .pth and .onnx files are stored here.
- `python_figures`: Output directory for all matplotlib figures generated within generate_model.py. Includes figures like loss curves, residuals, truth vs noisy input, noisy input vs model predictions, etc.
- `Plotting`: Directory containing [gnuplot-latex-utils](https://github.com/ksalamone59/gnuplot_latex_utils) as a submodule. Automatically stores output from C++ inference into the Plotting/ort_inference directory. The Makefile that exists here will automatically run the plotting scripts. Please see the original documentation for more information.
- `Makefile`: Runs the pipeline in order: from the python model generation and plot creation, to the C++ inference, to creating the final output. The input expected is `make ORT_DIR=/path/to/ONNXRunTime/`. Note: `make clean` will delete the model files as well for completeness. 

### How to Run the Code
- Run git submodule update --init --recursive
- Run `make ORT_DIR=/path/to/ONNXRunTime` from the base directory.
- The python output plots are stored in `python_figures` and the final C++ inference is stored in `Plotting/pdfs/ort_inference.pdf`

### Dependencies 
- For python3:
    - python3-dev (python3 >= 3.10)
    - numpy 
    - PyTorch 
    - tqdm
    - scipy
    - matplotlib
    - onnxruntime 
- ONNXRunTime C++ API, tested with 1.22.1
    - I installed it from [here](https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime/1.22.1)
- CMake
- At least c++17
- gnuplot and pdflatex (see [gnuplot-latex-utils](https://github.com/ksalamone59/gnuplot_latex_utils) for details)