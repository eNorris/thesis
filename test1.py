import sys
import time

import threading

import numpy
import matplotlib.pyplot as pyplot

#x = numpy.linspace(0, 1, 100)
#y = numpy.linspace(0, 1, 100)
#z = numpy.linspace(0, 1, 100)

#q = numpy.polynomial.legendre.leggauss(5)

def get_x_bins():
    return numpy.linspace(0, 1, 11)
    
    
def get_y_bins():
    return numpy.linspace(0, 1, 11)
    
    
def get_z_bins():
    return numpy.linspace(0, 1, 11)
    
    
def get_quad():
    directions = [[0,0,1], [0,1,0], [1,0,0]]
    wts = [1.0/3, 1.0/3, 1.0/3]
    
    return directions, wts
    
    
def get_e_bins():
    return numpy.logspace(-6, 2, 11)
    
    
def global_mesh_indx_to_r(indx):
    
    return 0, 0, 0
    
    
def get_mat_id(x, y, z):
    return 1
    
    
class MyThread(threading.Thread):
    def __init__(self, jumpahead, totalthreads, datasize, data):
        threading.Thread.__init__(self)
        self.jumpahead = jumpahead
        self.totalthreads = totalthreads
        self.datasize = datasize
        self.data = data
    def run(self):
        for i in range(self.datasize/self.totalthreads):
            self.data[i + self.jumpahead*self.datasize/self.totalthreads] = i + self.jumpahead*self.datasize/self.totalthreads
        print('finished')
    
xbins = get_x_bins();
ybins = get_y_bins();
zbins = get_z_bins();
ebins = get_e_bins();
quad = get_quad();

pyplot.close('all')

pyplot.plot(ebins)

pyplot.show()

mesh_size = len(xbins) * len(ybins) * len(zbins)
phase_size = mesh_size * len(ebins) * len(quad)

phase_bytes = 8*phase_size
phase_kb = numpy.ceil(phase_bytes / 1024)
phase_mb = numpy.ceil(phase_bytes / 1024**2)
phase_gb = numpy.ceil(phase_bytes / 1024**3)

if phase_gb > 16:
    print('Tried to allocate too much memory! (' + str(phase_gb) + "GB)")
    sys.exit(555)
    
if phase_gb > 0:
    print('Allocating ~' + str(phase_gb) + 'GB...')
if phase_mb > 0:
    print('Allocating ~' + str(phase_mb) + 'MB...')
if phase_kb > 0:
    print('Allocating ~' + str(phase_kb) + 'KB...')
print('Allocating ' + str(phase_bytes) + 'B...')


phase = numpy.zeros((phase_size))

phase[0] = 5
phase[phase_size-1] = 5

t1 = MyThread(0, 2, phase_size, phase)
t2 = MyThread(1, 2, phase_size, phase)

t1.start()
t2.start()

#for i in range(phase_size):
#    phase[i] = i

time.sleep(3)

mesh = numpy.zeros((mesh_size))
    
for i in range(mesh_size):
    x, y, z = global_mesh_indx_to_r(i)
    mat_id = get_mat_id(x, y, z)
    mesh[i] = mat_id

#del phase
#del mesh














