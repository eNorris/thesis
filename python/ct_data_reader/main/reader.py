import numpy
import matplotlib.pyplot as pyplot

__author__ = 'etnc6d'

def parse(filename):
    with open(filename, 'rb') as f:
        a = numpy.fromfile(f, dtype=numpy.uint16)

        return a

def write(filename, mtrx):
    print("Writing a matrix of size: " + str(mtrx.shape))
    with open(filename, 'wb') as f:
        f.write(mtrx.tobytes())
        #numpy.ndarray.tofile(f, "")

def simplify(mtrx):
    outmtrx = numpy.zeros((16, 64, 64), dtype=numpy.uint16)

    print(outmtrx.shape)
    print(mtrx.shape)

    for ox in range(64):
        for oy in range(64):
            for oz in range(16):
                v = 0.0
                for ix in range(4):
                    for iy in range(4):
                        for iz in range(4):
                            v += mtrx[4*oz+iz, 4*oy+iy, 4*ox+ix]
                outmtrx[oz, oy, ox] = v / (4*4*4)

    return outmtrx


mtrx = parse("/media/Storage/thesis/doctors/data/liver_Bay52DDC_volume.bin")


#mtrx[mtrx > 256] = 256


print("Size should be: " + str(256*256*64))
print(mtrx.shape)

mtrx = numpy.reshape(mtrx, (64, 256, 256))
smp = simplify(mtrx)
write("/media/Storage/thesis/doctors/data/liver_Bay52DDC_volume_simple.bin", numpy.reshape(smp, (16*64*64)))


pyplot.figure()
pyplot.contourf(mtrx[:, :, 3], 64, cmap='viridis', interpolation='nearest')
pyplot.colorbar()
pyplot.xlabel('Z')
pyplot.ylabel('Y')
pyplot.title("ZY")

pyplot.figure()
pyplot.contourf(mtrx[:, 32, :], 64, cmap='viridis', interpolation='nearest')
pyplot.colorbar()
pyplot.xlabel('Z')
pyplot.ylabel('X')
pyplot.title("ZX")

pyplot.figure()
pyplot.imshow(numpy.log10(mtrx[8, :, :]), cmap='viridis', interpolation='nearest', clim=(0, 3.5))
pyplot.colorbar()
pyplot.xlabel('Y')
pyplot.ylabel('X')
pyplot.title("YX")

pyplot.figure()
pyplot.imshow(numpy.log10(smp[2, :, :]), cmap='viridis', interpolation='nearest', clim=(0, 3.5))
pyplot.colorbar()
pyplot.xlabel('Y')
pyplot.ylabel('X')
pyplot.title("Simplified YX")

pyplot.show()