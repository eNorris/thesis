import numpy
import matplotlib.pyplot as pyplot

__author__ = 'etnc6d'

def parse(filename):
    with open(filename, 'rb') as f:
        a = numpy.fromfile(f, dtype=numpy.uint16)

        return a


mtrx = parse("/media/Storage/thesis/doctors/water35_volume.bin")

mtrx[mtrx > 256] = 256


print("Size should be: " + str(256*256*64))
print(mtrx.shape)

mtrx = numpy.reshape(mtrx, (64, 256, 256))

pyplot.figure()
pyplot.contourf(mtrx[:, :, 15], 64, cmap='viridis', interpolation='nearest')
pyplot.colorbar()
pyplot.xlabel('Z')
pyplot.ylabel('Y')
pyplot.title("ZY")

pyplot.figure()
pyplot.contourf(mtrx[:, 128, :], 64, cmap='viridis', interpolation='nearest')
pyplot.colorbar()
pyplot.xlabel('Z')
pyplot.ylabel('X')
pyplot.title("ZX")

pyplot.figure()
pyplot.contourf(mtrx[32, :, :], 64, cmap='viridis', interpolation='nearest')
pyplot.colorbar()
pyplot.xlabel('Y')
pyplot.ylabel('X')
pyplot.title("YX")

pyplot.show()