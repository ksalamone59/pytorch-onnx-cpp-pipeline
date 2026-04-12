load "../style/term.gnu"
load "../style/default.gnu"

set out 'example.tex'

set title "Example Plot: Decaying Oscillations with Noise" font ",16"
set xlabel '$x$' font ",14"
set ylabel '$y$' font ",14"

set label 1 front at 0.5, -3.5 center '$f(x)=e^{-0.3x}\cos(2\pi x)$ + '
set label 2 front at 0.5, -4.2 center 'Gaussian Noise. Width: $0.1$'

set size ratio 1
unset key 

plot "example.dat" u 1:2 w l ls 2 notitle