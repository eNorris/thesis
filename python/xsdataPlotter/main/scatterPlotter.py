import matplotlib.pyplot as pyplot
import numpy
import random
import scipy.special
from mpl_toolkits.mplot3d import Axes3D

__author__ = 'etnc6d'

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = numpy.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = numpy.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = numpy.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def parse_matmsh3(filename):
    dfile = open(filename, 'r')

    lines = dfile.readlines()

    dims = int(lines[0])
    if not dims == 3:
        raise Exception("Can only parse 3D data meshes")

    dim1size = int(lines[1])
    dim2size = int(lines[2])
    dim3size = int(lines[3])

    lindx = 4

    dim1 = []
    dim2 = []
    dim3 = []
    data = []

    for i in range(dim1size):
        dim1.append(float(lines[lindx]))
        lindx += 1

    for i in range(dim2size):
        dim2.append(float(lines[lindx]))
        lindx += 1

    for i in range(dim3size):
        dim3.append(float(lines[lindx]))
        lindx += 1

    for i in range(dim1size * dim2size * dim3size):
        data.append(float(lines[lindx]))
        lindx += 1

    ndata = numpy.array(data)
    ndata = ndata.reshape((dim1size, dim2size, dim3size))

    return numpy.array(dim1), numpy.array(dim2), numpy.array(dim3), ndata


def diracdelta(x, y):
    if x == y:
        return 1.0
    return 0.0


def factorial(x):
    if x <= 2:
        return x
    return x * factorial(x-1)


def clm(l, m):
    return numpy.sqrt((2 - diracdelta(2, 0)) * factorial(l-abs(m)) / factorial(l+abs(m)))

e, eprime, nl, vals = parse_matmsh3("/media/Storage/thesis/python/xsdataPlotter/be9scatter504.dat")

#e = e[::-1]
#eprime = eprime[::-1]

sh = vals.shape
nls = sh[2]

minv = vals.min()
maxv = vals.max()

pyplot.close("all")

indexes = numpy.linspace(1, 19, 19)

for i in range(nls):
    pyplot.figure()
    ax = pyplot.contourf(e/1E6, eprime/1E6, vals[:, :, i], 64, cmap='viridis', vmin=minv, vmax=maxv)
    caxis = pyplot.colorbar()
    pyplot.contour(e/1E6, eprime/1E6, vals[:, :, i], [0], colors='k')
    #ax.set_yscale('log')
    #ax.set_xscale('log')
    pyplot.yscale('log')
    pyplot.xscale('log')
    pyplot.xlabel("E")
    pyplot.ylabel("E'")
    pyplot.title("Scatter XS, Nl = " + str(i))

eIndex = 30
eValue = e[eIndex]
eprimeIndex = 30
eprimeValue = e[eprimeIndex]

sigma_ls = vals[eIndex, eprimeIndex, :]

x = []
y = []
z = []
mu = []
sigma = []
angles = 1000

for i in range(angles):
    theta = random.random() * 2 * numpy.pi
    u = random.random() * 2 - 1
    xx = numpy.sqrt(1 - u**2) * numpy.cos(theta)
    x.append(xx)
    y.append(numpy.sqrt(1 - u**2) * numpy.sin(theta))
    z.append(u)

    mu.append(xx)

    q = [numpy.sqrt(1 - u**2) * numpy.cos(theta)] * len(sigma_ls)
    #pl = numpy.polynomial.legendre.Legendre(q)


    s = 0
    for l, sigma_l in enumerate(sigma_ls):
        plfun = scipy.special.legendre(l)
        s += ((2 * l + 1) / (4 * numpy.pi)) * sigma_l * plfun(xx)

    sigma.append(s)

fig = pyplot.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot([0, 2.0], [0, 0], [0, 0])
ax.scatter(x, y, z, c=sigma, cmap='jet')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title("3-D Group " + str(eIndex) + "(" + str(e[eIndex]/1E6) + " - " + str(e[eIndex+1]/1E6) +
             "MeV )$\\rightarrow$ Group " + str(eprimeIndex) + "(" + str(e[eprimeIndex]/1E6) + " - " + str(e[eprimeIndex+1]/1E6) + " MeV)")
set_axes_equal(ax)

#mu1d = numpy.linspace(-1.0, 1.0, 1000)
mu1d = [1.0]
sigma1d = []
for i in range(len(mu1d)):
    s = 0
    p = []
    pp = []
    peval = []
    pl = []
    for l, sigma_l in enumerate(sigma_ls[:]):
        plfun = scipy.special.legendre(l)
        p.append(mu1d[i])
        peval.append(plfun(mu1d[i]))
        pp.append(((2 * l + 1) / (4 * numpy.pi)) * sigma_l * plfun(mu1d[i]))
        pl.append(sigma_l)
        s += ((2 * l + 1) / (4 * numpy.pi)) * sigma_l * plfun(mu1d[i])
    sigma1d.append(s)

    print(s)

pyplot.figure()
pyplot.plot(mu1d, sigma1d, 'b', [-1, 1], [0, 0], 'k--')
pyplot.xlabel('$\\mu$')
pyplot.ylabel('$\\sigma(\\mu)$')
pyplot.title("1-D Group " + str(eIndex) + "(" + str(e[eIndex]/1E6) + " - " + str(e[eIndex+1]/1E6) +
             "MeV )$\\rightarrow$ Group " + str(eprimeIndex) + "(" + str(e[eprimeIndex]/1E6) + " - " + str(e[eprimeIndex+1]/1E6) + " MeV)")

pyplot.show()