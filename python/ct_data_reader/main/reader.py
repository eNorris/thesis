import numpy
import matplotlib
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

font = {'family': 'Times New Roman', 'size': 16}
matplotlib.rc('font', **font)

mtrx = parse("/media/Storage/thesis/doctors/data/water35_volume.bin")


#mtrx[mtrx > 256] = 256


print("Size should be: " + str(256*256*64))
print(mtrx.shape)

#mtrx[mtrx==65535] = 0
artifact_count = sum([1 if x>=65500 else 0 for x in mtrx])
mtrx[:] = [0 if x>=65500 else x for x in mtrx]
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
hist, bin_edges = numpy.histogram(mtrx, 250)
hx = numpy.repeat(bin_edges, 2)
hx = hx[1:-1]
hy = numpy.repeat(hist, 2)
#hx = [bin_edges[0], numpy.repeat(bin_edges[1:-1], 2), bin_edges[-1]]
#hy = [h, h for h in hist]

pyplot.figure()
pyplot.plot(hx, hy)
pyplot.xlabel('CT Number')
pyplot.ylabel('Frequency')

pyplot.figure()
pyplot.semilogy(hx, hy)
pyplot.xlabel('CT Number')
pyplot.ylabel('Frequency')
#pyplot.title("Histogram")

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

slice = mtrx[32, :, :]
lineout = slice[:, 120]
pyplot.figure()
pyplot.plot(lineout)
pyplot.xlim([0, 255])
pyplot.xlabel('x')
pyplot.ylabel('Intensity')
#pyplot.title("Lineout")

pyplot.show()