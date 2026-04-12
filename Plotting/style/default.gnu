wi=5
# color blind safe palette from https://jfly.uni-koeln.de/color/
set style line 1 pt 7 lw wi lc "black"  
set style line 2 pt 3 lw wi lc "#56b4e9"
set style line 3 pt 5 lw wi lc "#d55e00"
set style line 4 lw wi lc "#0072b2"
set style line 5 lw wi lc "#f0e442"
set style line 6 dt 3 lw wi lc "black"  
set style line 7 dt 3 lw wi lc "#56b4e9"
set style line 8 dt 3 lw wi lc "#d55e00"
set style line 9 dt 3 lw wi lc "#0072b2"
set style line 10 dt 3 lw wi lc "#f0e442"
set style line 101  lw wi lc "#00c000"

set style line 22 pt 3 lw wi ps 2 lc "#56b4e9"

#for filling
set style line 11 lw 1 lc "black"  
set style line 12 lw 1 lc "#56b4e9"
set style line 13 lw 1 lc "#d55e00"
set style line 14 lw 1 lc "#0072b2"
set style line 15 lw 1 lc "#f0e442"
set style line 16 pt 5 lw wi lc "red"

set palette defined (0 "white",\
                     0.2 "purple",\
                     0.4 "blue",\
                     0.6 "green",\
                     0.8 "yellow",\
                     1.0 "orange",\
                     1.2 "red")

set style fill  transparent solid 0.5 noborder 

set key reverse Left
set tics format "$%g$" front

set xtics nomirror
set bars 3.0