__author__ = 'etnc6d'

import h5py
import numpy as np
import matplotlib.pyplot as pyplot

filename = '/media/Storage/advantage/C831MNYCP00/Source/Scale/Exnihilo/packages/Transcore/xslib/test/3mat.h5'

with h5py.File(filename, 'r') as hf:
    #print("List of arrays in this file:\n", hf.keys())

    print(hf.items())
    data = hf.get('mat1')
    print(data.items())
    dataa = data.get('P0')
    print(dataa)
    #print(data)
    #print(data.items())
    np_data = np.array(data)
    #print("Shape: ", np_data.shape)

    gdata = dataa[27:, 27:]
    print(gdata.shape)
    print(gdata[-1,-1])

    pyplot.figure()
    pyplot.contourf(gdata, 64)
    pyplot.show()