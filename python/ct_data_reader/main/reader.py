import numpy
import matplotlib
import matplotlib.pyplot as pyplot
import matplotlib.font_manager

__author__ = 'etnc6d'


def parse(filename):
    with open(filename, 'rb') as f:
        a = numpy.fromfile(f, dtype=numpy.uint16)

        return a


def write(filename, mtrx):
    print("Writing a matrix of size: " + str(mtrx.shape))
    with open(filename, 'wb') as f:
        f.write(mtrx.tobytes())


def simplify(matrix):
    out_matrix = numpy.zeros((16, 64, 64))  # , dtype=numpy.uint16)  # Setting the dtype is required for saving

    # print(out_matrix.shape)
    # print(matrix.shape)

    for ox in range(64):
        for oy in range(64):
            for oz in range(16):
                v = 0.0
                for ix in range(4):
                    for iy in range(4):
                        for iz in range(4):
                            v += matrix[4*oz+iz, 4*oy+iy, 4*ox+ix]
                out_matrix[oz, oy, ox] = v / (4*4*4)

    return out_matrix

# Set the font
font = {'family': 'Times New Roman', 'size': 16}
matplotlib.rc('font', **font)

# Fixes issue where Times New Roman is bold by default
del matplotlib.font_manager.weight_dict['roman']
matplotlib.font_manager._rebuild()

mtrx = parse("/media/Storage/thesis/doctors/data/water35_volume.bin")

print("Size should be: " + str(256*256*64))
print(mtrx.shape)

artifact_count = sum([1 if x >= 65500 else 0 for x in mtrx])
mtrx[:] = [0 if x >= 65500 else x for x in mtrx]

offset = 1024
mtrx = [x-offset for x in mtrx]

print("Artifacts: " + str(artifact_count))
'''
air_count = sum([1 if x < 67 else 0 for x in mtrx])
arti_count = sum([1 if 67 <= x < 600 else 0 for x in mtrx])
water_count = sum([1 if 600 <= x < 1080 else 0 for x in mtrx])
cont_count = sum([1 if 1080 <= x else 0 for x in mtrx])
total = 256*256*64.0

print("Water: " + str(water_count))
print("Container: " + str(cont_count))
print("Air: " + str(air_count))
print("Artifacts: " + str(arti_count))

print("Water: " + str(water_count/total))
print("Container: " + str(cont_count/total))
print("Air: " + str(air_count/total))
print("Artifacts: " + str(arti_count/total))

totalfound = air_count + arti_count + water_count + cont_count
print("Total identified: " + str(totalfound/total))
'''

hist, bin_edges = numpy.histogram(mtrx, 256)
hx = numpy.repeat(bin_edges, 2)
hx = hx[1:-1]
hy = numpy.repeat(hist, 2)

pyplot.figure()
pyplot.plot(hx, hy)
pyplot.xlabel('CT Number')
pyplot.ylabel('Frequency')
pyplot.hold(True)
pyplot.plot([0, 0], [0, pyplot.ylim()[1]], 'k--')

pyplot.figure()
pyplot.semilogy(hx, hy, [0, 0], [1, 1e6], 'k--')
pyplot.xlabel('CT Number')
pyplot.ylabel('Frequency')

mtrx = numpy.reshape(mtrx, (64, 256, 256))
smp = simplify(mtrx)
#write("/media/Storage/thesis/doctors/data/water35_volume_simple.bin", numpy.reshape(smp, (16*64*64)))


pyplot.figure()
pyplot.imshow(mtrx[:, :, 64], cmap='viridis', interpolation='nearest')
pyplot.colorbar()
pyplot.xlabel('$y$ index')
pyplot.ylabel('$z$ index')
pyplot.title("$x$ index = 64")

pyplot.figure()
pyplot.imshow(mtrx[:, 128, :], cmap='viridis', interpolation='nearest')
pyplot.colorbar()
pyplot.xlabel('$x$ index')
pyplot.ylabel('$z$ index')
pyplot.title("$y$ index = 128")

pyplot.figure()
pyplot.imshow(mtrx[32, :, :], cmap='viridis', interpolation='nearest')
pyplot.colorbar()
pyplot.xlabel('$x$ index')
pyplot.ylabel('$y$ index')
pyplot.title("$z$ index = 32")


pyplot.figure()
pyplot.imshow(smp[:, :, 16], cmap='viridis', interpolation='nearest')
pyplot.colorbar()
pyplot.xlabel('$y$ index')
pyplot.ylabel('$z$ index')
pyplot.title("$x$ index = 16")

pyplot.figure()
pyplot.imshow(smp[:, 32, :], cmap='viridis', interpolation='nearest')
pyplot.colorbar()
pyplot.xlabel('$x$ index')
pyplot.ylabel('$z$ index')
pyplot.title("$y$ index = 32")

pyplot.figure()
pyplot.imshow(smp[8, :, :], cmap='viridis', interpolation='nearest')
pyplot.colorbar()
pyplot.xlabel('$x$ index')
pyplot.ylabel('$y$ index')
pyplot.title("$z$ index = 8")

'''
pyplot.figure()
pyplot.imshow(smp[8, :, :], cmap='viridis', interpolation='nearest')
pyplot.colorbar()
pyplot.xlabel('$x$ index')
pyplot.ylabel('$y$ index')
pyplot.title("Simplified XY")
'''

matrix_slice = mtrx[32, :, :]
line_out = matrix_slice[:, 120]
pyplot.figure()
pyplot.plot(line_out)
pyplot.xlim([0, 255])
pyplot.xlabel('x')
pyplot.ylabel('Intensity')
#pyplot.title("Lineout")

pyplot.show()