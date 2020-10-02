from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,cross_val_score
import numpy as np
import matplotlib.pyplot as plt
import sys
import sklearn.linear_model as skl
from sklearn.metrics import mean_squared_error,r2_score
from functions import beta_, R2, mean_squared_error,beta_r,X_D,FrankeFunction
from b import bootstrap

def cross_validation(k,x,y,z,degree,model,lambda_):
    MSE_fold = np.zeros(degree)
    R2_score = np.zeros(degree)
    score = np.zeros(degree)
    scaler = StandardScaler()
    deg = np.linspace(1,degree,degree)

    for i in range(1,degree+1):
        print("Degree = %.3f"%i)
        X = X_D(x,y,i)
        index_arr = np.arange(0,X.shape[0],1)

        scaler.fit(X)
        X_train_scaled = scaler.transform(X)
        X_train_scaled[:,0] = 1                   #setting the first column = 1 because standard scaler sets it to 0

        np.random.shuffle(index_arr)

        X_train_scaled = X_train_scaled[index_arr]
        z_train = z[index_arr]

        X_train_scaled = np.array(np.array_split(X_train_scaled,k))
        z_split = np.array(np.array_split(z_train,k))




        for j in range(0,k):

            index = np.ones(k,dtype=bool)    #making index into True True ... array

            index[j] = False
            X_test_fold = X_train_scaled[j]
            z_test_fold = z_split[j]
            X_train_fold = X_train_scaled[index]     #inserting all indexes of X_train_scaled that are True
            X_train_fold = np.reshape(X_train_fold,(X_train_fold.shape[0]*X_train_fold.shape[1],X_train_fold.shape[2]))

            z_train_fold = np.ravel(z_split[index])
            if model == "ols":
                beta_fold = beta_(X_train_fold,z_train_fold)
                z_tilde_fold = X_test_fold.dot(beta_fold)
            if model == "ridge":
                beta_fold = beta_r(X_train_fold,z_train_fold,lambda_)
                z_tilde_fold = X_test_fold.dot(beta_fold)
            if model == "lasso":
                clf_lasso = skl.Lasso(alpha = lambda_,fit_intercept = False).fit(X_train_fold,z_train_fold)
                z_tilde_fold = clf_lasso.predict(X_test_fold)

            R2_score[i-1] += R2(z_test_fold,z_tilde_fold,np.mean(z_test_fold))
            MSE_fold[i-1] += mean_squared_error(z_test_fold,z_tilde_fold)

        MSE_fold[i-1]/=k
        R2_score[i-1]/=k


    return MSE_fold,R2_score,score,np.min(MSE_fold)


if __name__ == '__main__':
#np.random.seed(101)
    n = 50
    x = np.random.uniform(0,1,n)
    y = np.random.uniform(0,1,n)
    x = np.sort(x)
    y = np.sort(y)
    n = len(x)
    degree = 15
    deg = np.linspace(1,degree,degree)
    R2_score = np.zeros(degree)

    noise = 0.2*np.random.randn(n,n)
    #Its important to send the meshgrid into the design matrix function
    x,y = np.meshgrid(x,y)
    z =np.ravel((FrankeFunction(x,y)+noise))

    k = 5        #how many models we'll make   we have int(n/k) values per model

    MSE_fold,R2_score,score,min_error = cross_validation(k,x,y,z,degree,"ols",0)

    """
    plt.plot(deg,(MSE_fold),label="MSE")
    plt.xlabel("Complexity")
    plt.ylabel("Error")
    plt.legend()
    plt.show()




    plt.plot(deg,R2_score,label="manual k fold R2 score")
    plt.legend()
    plt.show()

    #Comparing with Bootstrap
    """

    B = 100
    _,MSE_boot,_,_,_ = bootstrap(B,x,y,z,"ols",0,degree)

    plt.title("Bootstrap vs Cross validation {:} folds {:} bootstraps ".format(k,B))
    plt.plot(deg,MSE_fold,label="Cross validation")
    plt.plot(deg,MSE_boot,label="Bootstrap")
    plt.xlabel("Complexity")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()
