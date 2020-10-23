import numpy as np
from numpy import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from functions import FrankeFunction,X_D
from sklearn.utils import resample
import matplotlib.pyplot as plt

n = 100
noise = 0.1
x = np.random.uniform(0,1,n)
y = np.random.uniform(0,1,n)
x,y = np.meshgrid(x,y) + noise*np.random.randn(n,n)
z = np.ravel(FrankeFunction(x,y))

class NN:
    def __init__(self, x, y, z, degree):
        self.x, self.y = x, y
        self.z = np.ravel(z)
        self.p = degree

    def X_D(self):
        #if the x, and y values are matrices we need to make them into vectors
        if len(self.x.shape)>1:
            x = np.ravel(self.x)    #flatten x
            y = np.ravel(self.y)
        self.N = len(x)
        l = int((self.p+1)*(self.p+2)/2) #number of columns in beta

        X = np.ones((self.N,l))

        for i in range(1,self.p+1):     #looping over the degrees but we have 21 elements
            q = int(i*(i+1)/2)           #getting the odd numbers 1.3.5 etc.
            for k in range(i+1):    #this loop makes sure we dont fill the same values in X over and over
                X[:,q+k] = x**(i-k)*y**k   #using the binomial theorem to get the right exponents
        self.D = X

    def train_test_split_scale(self, X, z):
        X_train,X_test,z_train,z_test = train_test_split(X,z,test_size=0.3)
        scaler = StandardScaler()
        scaler.fit(X_train)
        #X_train_scaled =  scaler.transform(X_train)
        #X_test_scaled = scaler.transform(X_test)
        return X_train, X_test, z_train, z_test

    def SGD(self, epochs, mini_batch_size, t0, t1, gamma=0, lam=0):
        X_train,X_test,z_train,z_test = self.train_test_split_scale(self.D,self.z)
        m = len(z_train)
        d = self.D.shape[1]
        weights = random.randn(d)
        def learning_schedule(t):
            return 0.01#/(1+0.000005*t)
        MSE_array = np.ones(epochs)*np.nan
        ind = np.arange(0,m)
        v = 0
        for i in range(epochs):
            random.shuffle(ind)
            X_train_shuffle = X_train[ind]
            z_train_shuffle = z_train[ind]
            for j in range(0,m,mini_batch_size):
                z_tilde = X_train_shuffle[j:j+mini_batch_size] @ weights
                diff_vec = z_tilde - z_train_shuffle[j:j+mini_batch_size]
                gradient = 2*(X_train_shuffle[j:j+mini_batch_size].T @ diff_vec)/mini_batch_size + 2*lam*weights
                eta = learning_schedule(i*m+j)
                v = gamma*v + eta*gradient
                weights = weights - v
            z_pred = X_train@weights
            MSE_array[i] = mean_squared_error(z_train, z_pred)
        MSE = mean_squared_error(z_train, z_pred)
        plt.plot(MSE_array, '.')
        plt.xlabel("epoch")
        plt.ylabel("MSE")
        plt.show()
        print(z_pred)
        print(z_train)
        return weights, MSE

    def OLS(self):
        X_train,X_test,z_train,z_test = self.train_test_split_scale(self.D,self.z)
        weights = np.linalg.pinv(X_train.T.dot(X_train)).dot(X_train.T).dot(z_train)
        z_pred = X_train@weights
        MSE = mean_squared_error(z_train, z_pred)
        return weights, MSE


my_instance = NN(x, y, z, 10)
my_instance.X_D()
weights, MSE = my_instance.SGD(200, 30, 5, 1, gamma = 0.9, lam = 0)
print(MSE)

weights_OLS, MSE_OLS = my_instance.OLS()
print(MSE_OLS)
