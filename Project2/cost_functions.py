import numpy as np

class MSE:
    """
    Mean squared error cost function
    """
    def __call__(self, y, a):
        return 1/len(y)*np.sum((a - y)**2)

    def deriv(self, y, a):
        return 2/len(y)*(a - y)

class accuracy:
    """
    Returns % of success in predictions. Only a measure
    """
    def __call__(self, y, a):
        return np.mean(y == a)

    def deriv(self):
        raise AttributeError("accuracy is only a measure and does not have a\
            derivative. Try a different cost function.")

class CE:
    """
    Log cross-entropy cost function.
    """
    def __call__(self,y,a):
        tmp1 = np.pow(a,y)
        tmp2 = np.prod(tmp1)
        return -np.log(tmp2)

    def deriv(self,y,a):
        return a-y
