load "../style/term.gnu"
load "../style/default.gnu"

set out "ort_inference.tex"
set lmargin 12
set ytics nomirror

set xlabel '$x$'
set ylabel '$f(x)=\sin(x)e^{-0.1x^{2}}$' offset -2,0
set title "ONNX RunTime C++ Inference"

set yrange[-1.25:1.25]

plot "cpp_input_data.dat" w p ls 2 ps 0.5 title 'Input Data',\
    "ort_inference.dat" with p ls 1 ps 0.5 title 'Inference from C++',\
    "real_function.dat" w l ls 3 dt (2,2) title 'Real Function',\
    keyentry title 'Inference $R^{2}$ = 0.998'
