import numpy as np 
x0, xf = -5, 5
step_size = 0.01
x = np.linspace(x0, xf, int((xf - x0) / step_size) + 1)
y = np.exp(-0.3 * x) * np.cos(2 * np.pi * x) + np.random.normal(0, 0.1, size=x.shape)
with open("example.dat", "w") as f:
    for i in range(len(x)):
        f.write(f"{x[i]} {y[i]}\n")