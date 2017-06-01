# -*- coding: utf-8 -*-
"""
The goal of this file is to compare fractional and normal Neural Networks

@author: Sharjeel Abid Butt

@References

1. 
"""

#import mnist_load as mload
#import copy
import dill

import numpy as np
#import scipy as sp
#import scipy.stats as stats
import pandas as pd
#import matplotlib as mpl
import matplotlib.pyplot as plt
#from scipy.special import gamma

def tic():
    #Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print "Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds."
    else:
        print "Toc: start time not set"

# Main Code starts here
filename= 'nnetCE.pkl'

#dill.dump_session(filename)

# and to load the session again:

dill.load_session(filename)

#ind1 = 8
#plt.figure()
#plt.plot(nnClassifier.eV[ind1], label = 'fr = 1')
#plt.hold(True)
#
#for ind2 in range(len(fr)):
#    plt.plot(frClassifier[ind1].eV[ind2], label = 'fr = ' +  str(fr[ind2]))
#
#plt.hold(False)
#plt.title('Learning rate = ' + str(LRVec[ind1]))
#plt.legend()
#plt.show()
plt.close('all')
for ind1 in range(len(fr)):
    plt.figure()
    plt.hold(True)
    
    for ind2 in range(len(LRVec)):
        plt.plot(nnClassifier.eV[ind2], label = 'NN LR = ' + str(LRVec[ind2]))
        plt.plot(frClassifier[ind2].eV[ind1], label = 'frNN LR = ' +  str(LRVec[ind2]))
        
    plt.hold(False)
    plt.legend()
    plt.title('fractional Order = ' + str(LRVec[ind1]))
    plt.show()
