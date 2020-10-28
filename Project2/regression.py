import numpy as np
from numpy import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from functions import FrankeFunction,X_D
from sklearn.utils import resample
import matplotlib.pyplot as plt
from data_prep import data_prep
n = 100
noise = 0.1
x = np.random.uniform(0,1,n)
y = np.random.uniform(0,1,n)
x,y = np.meshgrid(x,y) + noise*np.random.randn(n,n)
z = np.ravel(FrankeFunction(x,y))





class regression:
    def __init__(self, X_train, X_test, z_train, z_test):
        self.weights = None
        self.X_test = X_test
        self.X_train = X_train
        self.z_test = z_test
        self.z_train = z_train




    def SGD(self, epochs, mini_batch_size, t0, t1, gamma=0, lam=0):
        m = len(self.z_train)
        d = self.X_train.shape[1]
        self.weights = random.randn(d)
        def learning_schedule(t):
            return 0.0001#/(1+0.000005*t)
        MSE_array = np.ones(epochs)*np.nan
        ind = np.arange(0,m)
        v = 0
        for i in range(epochs):
            random.shuffle(ind)
            X_train_shuffle = self.X_train[ind]
            z_train_shuffle = self.z_train[ind]
            for j in range(0,m,mini_batch_size):
                z_tilde = X_train_shuffle[j:j+mini_batch_size] @ self.weights
                diff_vec = z_tilde - z_train_shuffle[j:j+mini_batch_size]
                gradient = 2*(X_train_shuffle[j:j+mini_batch_size].T @ diff_vec)/mini_batch_size + 2*lam*self.weights
                eta = learning_schedule(i*m+j)
                v = gamma*v + eta*gradient
                self.weights = self.weights - v
            z_pred = X_train@self.weights
            MSE_array[i] = mean_squared_error(self.z_train, z_pred)
        self.MSE = mean_squared_error(self.z_train, z_pred)
        plt.plot(MSE_array, '.')
        plt.xlabel("epoch")
        plt.ylabel("MSE")
        #plt.show()
        #print(z_pred)
        #print(self.z_train)

    def OLS(self):
        self.weights = np.linalg.pinv(self.X_train.T.dot(self.X_train)).dot(self.X_train.T).dot(self.z_train)
        z_pred = self.X_train@self.weights
        self.MSE = mean_squared_error(self.z_train, z_pred)




data = data_prep(x,y,z,10)
X_train, X_test, z_train, z_test = data()
my_instance = regression(X_train, X_test, z_train, z_test)

#doing SGD regression on our object
my_instance.SGD(200, 30, 5, 1, gamma = 0.9, lam = 0)
print("MSE from SGD:",my_instance.MSE)

#doing OLS regression on our object
my_instance.OLS()
print("MSE from OLS:",my_instance.MSE)
