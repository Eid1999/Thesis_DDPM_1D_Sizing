unset key
set grid
set title 'SSVC Amplifier (UMC_013): projection Fom vs. Gdc'
set xlabel 'Fom [MHz * pF / mA]'
set ylabel 'Gdc [Hz]'
plot '/home/nlourenco/.aida/plot-y1.dat' w points lw 1 pt 7 pointsize 1 lc rgb '#800000' 
pause mouse keypress
