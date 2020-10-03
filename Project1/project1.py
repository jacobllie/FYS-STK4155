import matplotlib.pyplot as plt
import numpy as np
from numpy import random
from franke_func import FrankeFunction
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

N = 50
sigma = 0.5

random.seed(11844)

#x = np.random.rand(N)
#y = np.random.rand(N)

x = np.linspace(0,1,N)
y = np.linspace(0,1,N)

X,Y = np.meshgrid(x,y)

z = FrankeFunction(X,Y)
z += sigma*random.randn(N,N)

indices = np.arange(0,N**2)
random.shuffle(indices)
z_shuffle = z.flatten()[indices]
x_shuffle = X.flatten()[indices]
y_shuffle = Y.flatten()[indices]

def create_X(x, y, n):
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    l = int((n+1)*(n+2)/2)          # Number of elements in beta
    X = np.ones((N,l))

    for i in range(1,n+1):
        q = int((i)*(i+1)/2)
        for k in range(i+1):
            X[:,q+k] = (x**(i-k))*(y**k)

    return X

def plot_franke():
    ax = fig.gca(projection='3d')
    fig = plt.figure()

    # Plot the surface.
    surf = ax.scatter(X, Y, z, s=3, c=z.flatten())
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

n = 20
train_error = np.zeros(n+1)
test_error = np.zeros(n+1)
MSE_kfold = np.zeros((n+1, 6))
#Setting up design matrix for polynomials of degree 5
for j in range(n+1):
    D = create_X(X,Y,j)
    #Finding polynomial coefficients
    beta = np.linalg.inv(D.T.dot(D)).dot(D.T).dot(np.ravel(z))
    #Predicted points
    z_tilde =  D@beta
    print("----------------------")
    print("Polynomial of degree",j)
    print("MSE: %.3e" %mean_squared_error(np.ravel(z), z_tilde))
    print("r2 score: %.3e" %r2_score(np.ravel(z), z_tilde))

    #Finding variance in coefficients
    var_beta = sigma**2*np.diagonal(np.linalg.inv(D.T.dot(D)))

    #Splitting data in train and test
    X_train, X_test, z_train, z_test = train_test_split(D, np.ravel(z), test_size=0.2)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    beta_train = np.linalg.pinv(X_train.T.dot(X_train)).dot(X_train.T).dot(np.ravel(z_train))
    #beta_scaled = np.linalg.pinv(X_train_scaled.T.dot(X_train_scaled)).dot(X_train_scaled.T).dot(np.ravel(z_train))

    z_train_tilde = X_train@beta_train
    z_test_tilde = X_test@beta_train

    #z_train_tilde = X_train_scaled@beta_scaled
    #z_test_tilde = X_test_scaled@beta_scaled

    train_error[j]= mean_squared_error(z_train, z_train_tilde)
    test_error[j] = mean_squared_error(z_test, z_test_tilde)

    for k in range(5,11):
        #Splitting data in k subsets
        N_cut = int(N**2/k)
        Z_k = np.zeros((k, N_cut))
        X_k = np.copy(Z_k)
        Y_k = np.copy(Z_k)
        for i in range(k):
            Z_k[i,:] = z_shuffle[int(i*N_cut):int((i+1)*N_cut)]
            X_k[i,:] = x_shuffle[int(i*N_cut):int((i+1)*N_cut)]
            Y_k[i,:] = y_shuffle[int(i*N_cut):int((i+1)*N_cut)]

        MSE = 0
        for l in range(k):
            ind_down = np.arange(l)
            ind_up = np.arange(l+1,k)
            ind = np.concatenate((ind_down, ind_up))
            D_l_train = create_X(X_k[ind],Y_k[ind],j)
            D_l_test = create_X(X_k[l],Y_k[l],j)
            beta_k = np.linalg.inv(D_l_train.T.dot(D_l_train)).dot(D_l_train.T).dot(np.ravel(Z_k[ind]))
            Z_l_test = D_l_test@beta_k

            MSE += mean_squared_error(Z_k[l], Z_l_test)

        MSE_kfold[j, k-5] = MSE/k

print("----------------------")
print("Mean value: %.2f" %np.mean(z))
print("----------------------")

"""plt.plot(train_error,label="Train")
plt.plot(test_error,label="Test")
plt.legend()
plt.ylabel("MSE")
plt.xlabel("Model complexity")
plt.show()"""

plt.plot(MSE_kfold)
plt.show()






























#
