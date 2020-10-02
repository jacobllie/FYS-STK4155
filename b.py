from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as skl
from functions import R2,mean_squared_error,beta_,beta_r,X_D,FrankeFunction

def bootstrap(B,x,y,z,model,lambda_,degree):
    train_bias = np.zeros(degree)
    test_bias = np.zeros(degree)
    train_variance = np.zeros(degree)
    test_variance = np.zeros(degree)
    train_error = np.zeros(degree)
    test_error = np.zeros(degree)
    deg = np.arange(0,degree,1)
    scaler = StandardScaler()

    for i in range(1,degree+1):
            print("Degree = %.3f"%i)
            X = X_D(x,y,i)               #X will have the same values on the columns
            X_train,X_test,z_train,z_test = train_test_split(X,np.reshape(z,(-1,1)),test_size=0.3)        #reshape the z because train test split does not accept arrays with shape (x,)
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train_scaled = scaler.transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            X_train_scaled[:,0] = 1                   #setting the first column = 1 because standard scaler sets it to 0
            X_test_scaled[:,0] = 1
            z_pred_train = np.zeros((z_train.shape[0],B))
            z_pred_test = np.zeros((z_test.shape[0],B))
            for j in range(B):
                index_arr = np.arange(0,len(np.ravel(z_train)),1)  #we will resample from the indexes of z, but without the test indexes

                tmp = resample(index_arr)
                X_ = X_train_scaled[tmp,:]        #filling an array with the random rows of the scaled design matrix
                z_ = z_train[tmp]
                if model == "ols":
                    beta_boot  = beta_(X_,z_)
                    z_pred_train[:,j] = np.ravel(X_train_scaled.dot(beta_boot))
                    z_pred_test[:,j] = np.ravel(X_test_scaled.dot(beta_boot))
                if model == "ridge":
                    beta_boot = beta_r(X_,z_,lambda_)
                    z_pred_train[:,j] = np.ravel(X_train_scaled.dot(beta_boot))
                    z_pred_test[:,j] = np.ravel(X_test_scaled.dot(beta_boot))
                if model == "lasso":
                    clf_lasso = skl.Lasso(alpha=lambda_,fit_intercept = False).fit(X_,z_)
                    z_pred_test[:,j] = clf_lasso.predict(X_test_scaled)



            train_error[i-1] = np.mean(np.mean((z_train - z_pred_train)**2, axis=1, keepdims=True) )
            test_error[i-1] = np.mean(np.mean((z_test - z_pred_test)**2, axis=1, keepdims=True) )
            min_test_error = np.min(test_error)
            test_bias[i-1] = np.mean( (z_test - np.mean(z_pred_test, axis=1, keepdims=True))**2 )

            test_variance[i-1] = np.mean( np.var(z_pred_test, axis=1, keepdims=True) )


    #Bootstrap bias- variance trade off

    """
    plt.title("Mean squared error")
    plt.plot(deg,train_error,label="Train")
    plt.plot(deg,test_error,label="Test")
    #plt.plot(deg,test_variance,label="variance")
    #plt.plot(deg,test_bias,label="bias")
    plt.legend()
    plt.show()
    """

    return train_error,test_error,test_bias,test_variance,min_test_error

if __name__ == '__main__':
    n = 50
    print(n)
    np.random.seed(130)
    x = np.random.uniform(0,1,n)
    y = np.random.uniform(0,1,n)
    noise = 0.1 * np.random.randn(n,n)
    x = np.sort(x)
    y = np.sort(y)
    x,y = np.meshgrid(x,y)
    z = np.ravel(FrankeFunction(x,y)+noise)
    degree = 20
    deg = np.linspace(1,degree,degree)
    MSE = np.zeros(degree)
    R2_score = np.zeros(degree)


    #Its important to send the meshgrid into the design matrix function

    B = 75

    train_error,test_error,bias,variance,min_test_error = bootstrap(B,x,y,z,"ols",0,degree)


    plt.style.use("seaborn")
    plt.title("MSE for {:d} bootstraps n = {:d}".format(B,n))
    plt.xlabel("Complexity")
    plt.ylabel("MSE")
    plt.plot(deg,train_error,label="Train")
    plt.plot(deg,test_error,label="Test")
    plt.legend()
    plt.savefig("./figures/b_train_test.jpg",bbox_inches = 'tight',pad_inches = 0.1)
    plt.show()

    plt.title("Bias variance plot for B = {:d} n = {:d}".format(B,n))
    plt.xlabel("Complexity")
    plt.ylabel("error")
    plt.plot(deg,test_error,label="MSE")
    plt.plot(deg,bias,label="bias")
    plt.plot(deg,variance,label="variance")
    plt.legend()
    plt.savefig("./figures/b_bias_variance.jpg",bbox_inches = 'tight',pad_inches = 0.1)
    plt.show()
