import numpy
import matplotlib.pyplot as pyplot

__author__ = 'etnc6d'

raw_uncollided_flux = numpy.loadtxt('/media/Storage/thesis/build-doctors-Desktop_Qt_5_4_1_GCC_64bit-Debug/uncol_flux.dat')

uflux = numpy.reshape(raw_uncollided_flux, (19, 64, 64, 16))

pyplot.figure()
pyplot.contourf(numpy.log10(uflux[18, :, :, 8]), 128, cmap='viridis')
pyplot.colorbar()
pyplot.title('Collided Flux')
pyplot.xlabel('-y')
pyplot.ylabel('x')


pyplot.figure()
x = numpy.linspace(50./128, 50-50./128, 64)
#r = x - 25 - 50./128
#print(r)
#r2 = 1.0/(r**2)
#r2 *= (max(uflux[18, :, 4, 8])/max(r2))
pyplot.semilogy(x, uflux[18, :, 4, 8])
pyplot.legend(['DOCTORS', '1/r^2'])


pyplot.show()
