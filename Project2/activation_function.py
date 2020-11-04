import numpy as np

class sigmoid:
    """def __init__(self):       #it might need an init to work
        pass"""
    def __call__(self,z):
        return 1/(1+np.exp(-z))
    def deriv(self,z):
        return np.exp(-z)/(1+np.exp(-z))**2

class tanh:
    def __call__(self,z):
        return (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))
    def deriv(self,z):
        return 1-((np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z)))**2

class relu:
    def __call__(self,z):
        z = np.array(z)
        z_neg = np.where(z < 0)
        z[z_neg] = 0
        return z
    def deriv(self,z):
        z = np.array(z)
        z_neg = np.where(z < 0)
        z[:] = 1
        z[z_neg] = 0
        return z

class leaky_relu:
    def __call__(self, z, alpha=0.1):
        z = np.array(z)
        z_neg = np.where(z < 0)
        z[z_neg] = alpha*z[z_neg]
        return z
    def deriv(self, z, alpha=0.1):
        z = np.array(z)
        z_neg = np.where(z < 0)
        z[:] = 1
        z[z_neg] = alpha
        return z

class elu:
    def __call__(self,z,alpha):
        if z <= 0:
            return alpha*(np.exp(z)-1)
        else:
            return z
    def deriv(self,z):
        if z<=0:
            return alpha*(np.exp(z)-1) + alpha
        else:
            return np.ones(z.shape)

class identity:
    def __call__(self,z):
        return z

    def deriv(self,z):
        return np.ones(z.shape)

"""class softmax:
    def __call__(self,z):
        #sum = np.sum(np.exp(z), axis=1).reshape(-1,1)
        sum = np.sum(np.exp(z))

        return np.exp(z)/sum #dividing all the rows with the sum of its column.

    def deriv(self,z):
        #print(self.__call__(z)-(self.__call__(z))**2)
        return self.__call__(z)-(self.__call__(z))**2  #assume softmax-softmax**2 because we're on the output layer"""

class softmax:
    def __call__(self, z):
        return np.exp(z)/np.sum(np.exp(z), axis=1)[:, None]

    def deriv(self, z):
        return self.__call__(z) - (self.__call__(z))**2
        # z * np.ones(len(z)).T.dot(np.eye(len(z)) - np.ones(len(z))*z.T)
