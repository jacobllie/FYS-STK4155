import numpy as np

class sigmoid:
    """def __init__(self):       #it might need an init to work
        pass"""
    def __call__(self,z):
        return 1/(1+np.exp(-z))
    def deriv(self,z):
        return np.exp(-z)/(1+np.exp(-z)**2)

class tanh:
    def __call__(self,z):
        return (np.exp(weight_sum)-np.exp(-weight_sum))/(np.exp(weight_sum)+np.exp(-weight_sum))
    def deriv(self,z):
        return 1-((np.exp(weight_sum)-np.exp(-weight_sum))/(np.exp(weight_sum)+np.exp(-weight_sum)))**2

class relu:
    def call(self,z):
        if z <= 0:
            return 0
        else:
            return z
    def deriv(self,z):
        if z <= 0:
            return 0
        else:
            return 1

class leaky_relu:
    def __call__(self,z):
        if z <= 0:
            return 0.1*z
        else:
            return z

    def deriv(self,z):
        if z <= 0:
            return 0
        else:
            return 1
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
            return 1
