import numpy
import matplotlib.pyplot as pyplot

__author__ = 'etnc6d'

raw = numpy.loadtxt('/media/Storage/thesis/mcnp.gitignore/meshtal_data_unc.dat')
raw2 = numpy.loadtxt('/media/Storage/thesis/mcnp.gitignore/meshtal_data_tot.dat')

print("Finished reading...")

e = raw[:, 0]
x = raw[:, 1]
y = raw[:, 2]
z = raw[:, 3]
v = raw[:, 4]
r = raw[:, 5]

vtot = raw2[:, 4]
rtot = raw2[:, 5]

e = e.reshape((5, 64, 64, 16))
x = x.reshape((5, 64, 64, 16))
y = y.reshape((5, 64, 64, 16))
z = z.reshape((5, 64, 64, 16))
v = v.reshape((5, 64, 64, 16))
r = r.reshape((5, 64, 64, 16))

vtot = vtot.reshape((5, 64, 64, 16))
rtot = rtot.reshape((5, 64, 64, 16))

eaxis = e[:, 0, 0, 0]
xaxis = x[0, :, 0, 0]
yaxis = y[0, 0, :, 0]
zaxis = z[0, 0, 0, :]

eaxis = numpy.insert(eaxis, 0, 0.0)
elabels = [str(erg1) + " - " + str(erg2) + " MeV" for erg1, erg2 in zip(eaxis[:-1], eaxis[1:])]
elabels[-1] = 'Total'

zslice = 3

pyplot.close('all')


for eindx, erg in enumerate(elabels):

    if max(v[eindx, :, :, zslice].flatten()) <= 0:
        continue

    #pyplot.figure()
    #pyplot.contourf(xaxis, yaxis, v[eindx, :, :, zslice], 128, cmap='viridis')
    #pyplot.colorbar()
    #pyplot.title('Unc Flux, E = ' + str(erg))
    #pyplot.xlabel('x')
    #pyplot.ylabel('y')

    pyplot.figure()
    pyplot.contourf(xaxis, yaxis, r[eindx, :, :, zslice], 128, cmap='viridis')
    pyplot.colorbar()
    pyplot.contour(xaxis, yaxis, r[eindx, :, :, zslice], [.05, .1, .2])
    pyplot.title('Unc Uncertainty, E = ' + str(erg))
    pyplot.xlabel('x')
    pyplot.ylabel('y')

    logticks = numpy.linspace(-8, 0, 9, endpoint=True)
    loglvs = numpy.linspace(-8, 0, 128, endpoint=True)

    pyplot.figure()
    pyplot.contourf(numpy.log10(v[eindx, :, :, zslice]), levels=loglvs, cmap='viridis')
    pyplot.colorbar(ticks=logticks)
    pyplot.title('Unc Log Flux, E = ' + str(erg))
    pyplot.xlabel('-y')
    pyplot.ylabel('x')

    pyplot.figure()
    pyplot.contourf(numpy.log10(vtot[eindx, :, :, zslice]), levels=loglvs, cmap='viridis')
    pyplot.colorbar(ticks=logticks)
    pyplot.title('Tot Log Flux, E = ' + str(erg))
    pyplot.xlabel('-y')
    pyplot.ylabel('x')

    pyplot.figure()
    pyplot.contourf(v[eindx, :, :, zslice]/vtot[eindx, :, :, zslice], 128, cmap='viridis')
    pyplot.colorbar(ticks=numpy.linspace(0, 1.0, 11, endpoint=True))
    pyplot.title('Unc/Total, E = ' + str(erg))
    pyplot.xlabel('-y')
    pyplot.ylabel('x')

pyplot.show()

