.PHONY: all train infer plot clean
ORT_DIR ?= /usr/local/onnxruntime
all: train infer plot

train: model_files/function_model.onnx
model_files/function_model.onnx: python/generate_model.py 
	cd python && python3 generate_model.py

infer: Plotting/ort_inference/ort_inference.dat
Plotting/ort_inference/ort_inference.dat: cpp/ort_inference_from_model.cpp cpp/CMakeLists.txt model_files/function_model.onnx
	mkdir -p cpp/build
	cd cpp/build && cmake .. -DONNXRUNTIME_DIR=$(ORT_DIR) && make -j && ./onnx_inference

Plotting/pdfs/ort_inference.pdf: Plotting/ort_inference/ort_inference.dat Plotting/Makefile Plotting/ort_inference/ort_inference.gnu Plotting/style/default.gnu Plotting/style/term.gnu Plotting/fix.py
	cd Plotting && make
plot: Plotting/pdfs/ort_inference.pdf

clean:
	cd Plotting && make clean
	cd Plotting/ort_inference && rm -f *.dat
	[ -d cpp/build ] && cd cpp/build && make clean || true
	rm -f cpp/build/onnx_inference python_figures/*.pdf
	cd model_files && rm -f *