import sys, os, importlib
from importlib import reload

import numpy as np
import matplotlib.pyplot as plt

import constants as gv
importlib.reload(sys.modules['constants'])

for i in range(0,9):
    gv.folder = 'kappaVsJ0_%d' %i
    gv.init_param()
    
    overlap = np.loadtxt(gv.path + 'overlap.dat')
    print(overlap.shape)
    J0 = -overlap[:,0]
    kappas = overlap[:,1]

    print(J0)
    print(kappas)
    plt.plot(J0,kappas,'-o')
    plt.xlabel('$J_0$')
    plt.ylabel('$\kappa$')

plt.show()
