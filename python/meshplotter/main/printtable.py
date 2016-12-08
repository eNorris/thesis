import numpy
import matplotlib
import matplotlib.pyplot as pyplot

__author__ = 'etnc6d'

pyplot.close('all')
filename = "/media/Storage/thesis/mcnp.gitignore/printtable60.dat"
raw = numpy.loadtxt(filename)

rawtrue = numpy.loadtxt("/media/Storage/thesis/mcnp.gitignore/ctdensity.dat")

num = raw[:, 0]
cell = raw[:, 1]
mat = raw[:, 2]
atom = raw[:, 3]
den = raw[:, 4]
vol = raw[:, 5]
mass = raw[:, 6]
pieces = raw[:, 7]
imp = raw[:, 8]

print(raw.shape)

for matid in range(1, 20):
    mat1den = den[mat == matid]

    hist, bin_edges = numpy.histogram(mat1den, 100)
    bins = (bin_edges[:-1] + bin_edges[1:])/2

    pyplot.figure()
    pyplot.plot(bins, hist, 'k')
    pyplot.title("Mat " + str(matid) + " Density [g/cc] Distribution")



hist, bin_edges = numpy.histogram(den, 500)
bins = (bin_edges[:-1] + bin_edges[1:])/2
hist[hist == 0] = 1
pyplot.figure()
pyplot.semilogy(bins, hist, 'k')
pyplot.title("Total Density [g/cc] Distribution")


rawtrue[rawtrue > 2.5] = 0
hist2, bin_edges2 = numpy.histogram(rawtrue, 500)
bins2 = (bin_edges2[:-1] + bin_edges2[1:])/2
hist[hist == 0] = 1
pyplot.figure()
pyplot.semilogy(bins2, hist2, 'k')
pyplot.title("True Density [g/cc] Distribution")

pyplot.figure()
pyplot.semilogy(bins, hist, 'r', bins2, hist2, 'k')
pyplot.title("Density [g/cc] Distribution")
pyplot.legend(['Calculated', 'True'])

pyplot.show()