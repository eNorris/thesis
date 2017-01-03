import numpy
import matplotlib.pyplot as pyplot

__author__ = 'etnc6d'

raw_uncollided_flux = numpy.loadtxt('/media/Storage/thesis/build-doctors-Desktop_Qt_5_4_1_GCC_64bit-Debug/uncol_flux.dat')
raw_ext_src = numpy.loadtxt('/media/Storage/thesis/build-doctors-Desktop_Qt_5_4_1_GCC_64bit-Debug/externalSrc.dat')
raw_sol = numpy.loadtxt('/media/Storage/thesis/build-doctors-Desktop_Qt_5_4_1_GCC_64bit-Debug/solution.dat')

uflux = numpy.reshape(raw_uncollided_flux, (19, 64, 64, 16))
source = numpy.reshape(raw_ext_src, (19, 64, 64, 16))
soln = numpy.reshape(raw_sol, (19, 64, 64, 16))

pyplot.figure()
pyplot.contourf(numpy.log10(uflux[18, :, :, 8]), 128, cmap='viridis')
pyplot.colorbar()
pyplot.title('Uncollided Flux')
pyplot.xlabel('-y')
pyplot.ylabel('x')


pyplot.figure()
pyplot.contourf(numpy.log10(source[18, :, :, 8]), 128, cmap='viridis')
pyplot.colorbar()
pyplot.title('Uncollided Source')
pyplot.xlabel('-y')
pyplot.ylabel('x')

pyplot.figure()
pyplot.contourf(numpy.log10(soln[18, :, :, 8]), 128, cmap='viridis')
pyplot.colorbar()
pyplot.title('Total Flux')
pyplot.xlabel('-y')
pyplot.ylabel('x')

pyplot.figure()
pyplot.contourf(soln[18, :, :, 8], 128, cmap='viridis')
pyplot.colorbar()
pyplot.title('Linear Total Flux')
pyplot.xlabel('-y')
pyplot.ylabel('x')




pyplot.show()
