import numpy as np

def mean_value(z):
    """
    Function for computing the mean value of the data set
    """
    S=0
    z_mrk = np.ravel(z)
    for j in range(int(N**2)):
        S += z_mrk[j]
    return 1/N**2*S

def compute_R2(z, z_tilde, z_mean=None):
    """
    Function for computing R squared score
    """
    if z_mean == None:
        z_mean = mean_value(z)

    z_mrk = np.ravel(z)
    S1 = (z_mrk - z_tilde)**2
    S2 = (z_mrk - z_mean)**2

    return 1 - np.sum(S1)/np.sum(S2)


def compute_MSE(z, z_tilde):
    """
    Function for computing mean squared error
    """
    N_mrk = len(z)
    z_mrk = np.ravel(z)
    S = (z_mrk - z_tilde)**2
    return 1/N_mrk**2*np.sum(S)
