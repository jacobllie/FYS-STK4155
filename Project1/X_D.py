import numpy as np
from numba import jit

def X_D(x,y,p):
    #if the x, and y values are matrices we need to make them into vectors
    if len(x.shape)>1:
        x = np.ravel(x)    #flatten x
        y = np.ravel(y)

    N = len(x)
    l = int((p+1)*(p+2)/2) #number of columns in beta

    X = np.ones((N,l))



    for i in range(1,p+1):     #looping over the degrees but we have 21 elements
        q = int(i*(i+1)/2)           #getting the odd numbers 1.3.5 etc.
        for k in range(i+1):    #this loop makes sure we dont fill the same values in X over and over
            X[:,q+k] = x**(i-k)*y**k   #using the binomial theorem to get the right exponents

    return X
