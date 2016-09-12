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

def factorial(x):
    if x == 0:
        return 1.0
    if x <= 2:
        return x
    return x * factorial(x-1)


def plm(mu, l, m):
    return scipy.special.lpmv(m, l, mu)


#def ylm(theta, phi, l, m):
#    mu = numpy.cos(theta)
#    mphi = abs(m) * phi
#    real = numpy.sqrt((2*l + 1)/(4*numpy.pi) * factorial(l-m)/factorial(l+m)) * plm(mu, l, m) * numpy.cos(mphi)
#    img = numpy.sqrt((2*l + 1)/(4*numpy.pi) * factorial(l-m)/factorial(l+m)) * plm(mu, l, m) * numpy.sin(mphi)
#    return real, img


#def ylm_conj(theta, phi, l, m):
#    rl, im = ylm(theta, phi, l, m)
#    return rl, -im


def ylm_e(theta, phi, l, m):
    mu = numpy.cos(theta)
    c = (-1.0)**m * numpy.sqrt((2l+1)/(2*numpy.pi) * factorial(l-m)/factorial(l+m))
    return c * plm(mu, l, m) * numpy.cos(m * phi)


def ylm_o(theta, phi, l, m):
    mu = numpy.cos(theta)
    c = (-1.0)**m * numpy.sqrt((2l+1)/(2*numpy.pi) * factorial(l-m)/factorial(l+m))
    return c * plm(mu, l, m) * numpy.sin(m * phi)


def ylm_0(theta, phi, l, m):
    mu = numpy.cos(theta)
    c = numpy.sqrt((2l+1)/(4*numpy.pi) * factorial(l-m)/factorial(l+m))
    return c * plm(mu, l, m) * numpy.cos(m * phi)

pyplot.close("all")
e, eprime, nl, vals = parse_matmsh3("/media/Storage/thesis/python/xsdataPlotter/be9scatter504.dat")

sh = vals.shape
nls = sh[2]

minv = vals.min()
maxv = vals.max()

pyplot.close("all")

eIndex = 35
eValue = e[eIndex]
eprimeIndex = 35
eprimeValue = e[eprimeIndex]

sigma_ls = vals[eIndex, eprimeIndex, :]
sigma_ls = sigma_ls[::-1]

xs = []
ys = []
zs = []
thetas = []
phis = []
#mus = []
sigmas = []
#sigma_imgs = []
angles = 10000
ref_dir = (1, 0, 0)

ref_theta = numpy.pi/2
ref_phi = 0

for _ in range(angles):
    phi = random.random() * 2 * numpy.pi
    mu = random.random() * 2 - 1
    x = numpy.sqrt(1 - mu**2) * numpy.cos(phi)
    y = numpy.sqrt(1 - mu**2) * numpy.sin(phi)
    z = mu
    xs.append(x)
    ys.append(y)
    zs.append(z)

    theta = numpy.arccos(mu)  # [0, pi]
    thetas.append(theta)
    phis.append(phi)

    #mus.append(numpy.dot(ref_dir, (x, y, z)))

for i in range(angles):
    s = 0

    for l in range(len(sigma_ls)):
        sigma_l = sigma_ls[l]
        ylm_sum = ylm_0(thetas[i], phis[i], l, 0)*ylm_0(ref_theta, ref_phi, l, 0)
        for m in range(1, l+1):
            ylm_sum += ylm_e(thetas[i], phis[i], l, m)*ylm_e(ref_theta, ref_phi, l, m) + ylm_o(thetas[i], phis[i], l, m)*ylm_o(ref_theta, ref_phi, l, m)

        s += sigma_l * ylm_sum

    sigmas.append(s)

#q, r = numpy.mgrid[0:2*numpy.pi:20j, 0:numpy.pi:10j]
#X=numpy.cos(thetas)*numpy.sin(phis)
#Y=numpy.sin(thetas)*numpy.sin(phis)
#Z=numpy.cos(phis)

fig = pyplot.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot([0, 2*ref_dir[0]], [0, 2*ref_dir[1]], [0, 2*ref_dir[2]])
p = ax.scatter(xs, ys, zs, marker=".", lw=0, c=sigmas, cmap='viridis')
# X, Y, Z = numpy.meshgrid(xs, ys, zs)
#p = ax.plot_wireframe(X, Y, Z, color='r')
#p = ax.plot_surface(xs, ys, zs, color='r')
fig.colorbar(p)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title("3-D Group " + str(eIndex) + "(" + str(e[eIndex]/1E6) + " - " + str(e[eIndex+1]/1E6) +
             "MeV )$\\rightarrow$ Group " + str(eprimeIndex) + "(" + str(e[eprimeIndex]/1E6) + " - " + str(e[eprimeIndex+1]/1E6) + " MeV)")
set_axes_equal(ax)

#print("xs size = " + str(len(xs)))
#print("sig size = " + str(len(sigmas)))

# Comparison to a legendre poly
mu = numpy.linspace(-1, 1, 1000)
s1d = numpy.zeros(1000)
for i in range(len(mu)):
    s = 0

    for l in range(len(sigma_ls)):
        sigma_l = sigma_ls[l]
        plm = scipy.special.legendre(l)
        s += sigma_l * (2*l+1)/(4*numpy.pi) * plm(numpy.cos(mu[i]))

    s1d[i] = s


pyplot.figure()
pyplot.scatter(xs, sigmas)
pyplot.plot(mu, s1d)
pyplot.plot([-1, 1], [0, 0], 'k--')
pyplot.xlabel('$\\mu$')
pyplot.ylabel('$\\sigma(\\mu)$')
pyplot.title("1-D Group " + str(eIndex) + "(" + str(e[eIndex]/1E6) + " - " + str(e[eIndex+1]/1E6) +
             "MeV )$\\rightarrow$ Group " + str(eprimeIndex) + "(" + str(e[eprimeIndex]/1E6) + " - " + str(e[eprimeIndex+1]/1E6) + " MeV)")


pyplot.show()