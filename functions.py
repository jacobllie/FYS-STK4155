import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4
def var(X,noise):

    variance = np.diag(noise*np.linalg.pinv(X.T.dot(X)))

    return variance    #We dont have to find the variance of the noise, we only have to multiply with the weight of the noise


def beta_(X,z):     #We have made the function usable for both scaled and non scaled design matrix
        beta = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(z)
        return beta


def mean_squared_error(y,y_tilde):     #sending in raveled z and z_tilde
    MSE = 1/len(y)*np.sum((y-y_tilde)**2)
    return MSE

def R2(y,y_tilde,y_mean):
    R2_score = 1-np.sum((y-y_tilde)**2)/np.sum((y-y_mean)**2)
    return R2_score

def beta_r(X,z,lambda_):
    I = X.shape[1]
    beta_ridge = np.linalg.pinv(X.T.dot(X)+lambda_*np.identity(I)).dot(X.T).dot(z)
    return beta_ridge

def terrain_data(filename):
    # Load the terrain
    terrain1 = imread(filename)

    # Show the terrain
    """
    plt.figure()
    plt.title("Kamloops")
    plt.imshow(terrain1, cmap="gray")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()"""
    return terrain1


def contour_plot(x,y,z):
    fig,ax=plt.subplots(1,1)
    cp = ax.contour(x, y, z)
    fig.colorbar(cp) # Add a colorbar to a plot
    ax.set_title('Filled Contours Plot')
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    plt.show()
