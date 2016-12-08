import numpy
import matplotlib
import matplotlib.pyplot as pyplot

__author__ = 'etnc6d'

filename = "/media/Storage/thesis/mcnp.gitignore/meshdata.dat"

raw = numpy.loadtxt(filename)

e = raw[:, 0]
x = raw[:, 1]
y = raw[:, 2]
z = raw[:, 3]
v = raw[:, 4]
r = raw[:, 5]

print(raw.shape)

newshape = (65, 65, 17)
e = numpy.reshape(e, newshape)
x = numpy.reshape(x, newshape)
y = numpy.reshape(y, newshape)
z = numpy.reshape(z, newshape)
v = numpy.reshape(v, newshape)
r = numpy.reshape(r, newshape)

#sm = sum(v)
#print("sum = " + str(sm))

pyplot.close('all')
matplotlib.rcParams['contour.negative_linestyle'] = 'solid'

pyplot.figure()
pyplot.contourf(x[:, :, 8], y[:, :, 8], v[:, :, 8], 64, cmap='viridis')
pyplot.colorbar()
pyplot.title("flux")

pyplot.figure()
cs = pyplot.contourf(x[:, :, 8], y[:, :, 8], numpy.log10(v[:, :, 8]), 256, cmap='viridis')
pyplot.colorbar()
pyplot.contour(cs, colors='k', hold='on', levels=numpy.log10([.01, .001, .0001]))
pyplot.title("log10(flux)")

pyplot.figure()
cs = pyplot.contourf(x[:, :, 8], y[:, :, 8], numpy.log10(r[:, :, 8]), 256, cmap='viridis')
pyplot.colorbar()
pyplot.title("log10(Uncertainty)")
pyplot.contour(cs, colors='k', hold='on', levels=numpy.log10([.1, .2, .3]))

pyplot.figure()
cs = pyplot.contourf(x[:, :, 8], y[:, :, 8], r[:, :, 8], 256, cmap='viridis')
pyplot.colorbar()
pyplot.title("Uncertainty")
cs2 = pyplot.contour(cs, colors='k', hold='on', levels=[.02, .05, .1, .2, .3])
pyplot.clabel(cs2, inline=1, fontsize=10)

pyplot.show()