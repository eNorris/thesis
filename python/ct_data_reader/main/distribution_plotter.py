import numpy
import matplotlib.pyplot as pyplot

__author__ = 'etnc6d'

with open("/media/Storage/thesis/build-doctors-Desktop_Qt_5_4_1_GCC_64bit-Debug/ctdist.dat", 'r') as f:
    a = numpy.loadtxt(f)

x = a[:-1, 0]
v = a[:-1, 1]
shift = 1024
x -= shift
overflow = a[-1, 1]

pyplot.figure()
pyplot.semilogy(x, v, 'r', [0, 0], [1, 1E5], 'k--', [-1000, -1000], [1, 1E5], 'k--')
pyplot.title("HU Frequency (Liver)")
pyplot.xlabel("Houndsfield Number")
pyplot.ylabel("Count")
pyplot.axis([-shift, 3000-shift, 0, 100000])

print("Overflows: " + str(overflow))

v = pyplot.cm.viridis(numpy.linspace(0, 256, 256, dtype=numpy.int))
r = [x[0] for x in v]
g = [x[1] for x in v]
b = [x[2] for x in v]

for r, g, b in zip(r, g, b):
    # brushes.push_back(QBrush(QColor::fromRgbF(0.0000,    1.0000,    1.0000)));
    s = "brushes.push_back(QBrush(QColor::fromRgbF(" + str(r) + ",    " + str(g) + ",    " + str(b) + ")));"
    print(s)

#print(pyplot.cm.viridis(numpy.linspace(0, 256, 256, dtype=numpy.int)))

pyplot.show()