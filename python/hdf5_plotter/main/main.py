__author__ = 'etnc6d'

import h5py
import numpy as np
import matplotlib.pyplot as pyplot

filename = '/media/Storage/advantage/C831MNYCP00/Source/Scale/Exnihilo/packages/Transcore/xslib/test/3mat.h5'

with h5py.File(filename, 'r') as hf:
    #print("List of arrays in this file:\n", hf.keys())

    print(hf.items())
    data = hf.get('mat2')
    print(data.items())
    dataa = data.get('total')
    scatdata = data.get('P0')
    gscatdata = scatdata[27:, 27:]
    print(dataa)
    #print(data)
    #print(data.items())
    np_data = np.array(data)
    #print("Shape: ", np_data.shape)

    gdata = dataa[27:]
    print(gdata.shape)
    print(gdata[-1])

    print("18, 18: " + str(gdata[18]))
    print("0, 0: " + str(gdata[0]))
    print("Ratio: " + str(gdata[0]/gdata[18]))

    for i in range(19):
        ln = "\t".join([str(x) for x in gscatdata[i,:]])
        print(ln)
    #print(gscatdata)

    for i in range(19):
        ln = "\t".join(["1 " if x != 0 else "0 " for x in gscatdata[i,:]])
        print(ln)

    pyplot.figure()
    pyplot.semilogy(gdata)

    pyplot.figure()
    pyplot.contourf(gscatdata, 64)
    pyplot.colorbar()
    pyplot.title("True $\\sigma_0$")

    pyplot.figure()
    pyplot.hist(gscatdata.tolist(), 50, facecolor='green')


    pyplot.show()