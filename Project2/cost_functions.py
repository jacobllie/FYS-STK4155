import numpy as np

class MSE:
    def __call__(self, y, a):
        return 1/len(y)*np.sum(a - y)**2

    def deriv(self, y, a):
        return 2/len(y)*(a - y)

class accuracy:
    def __call__(self,y,a):
        I = [1 for i in range(len(y)) if y[i]==a[i]]
        return np.sum(I)/len(y)
    def deriv(self):
        return 0.01
