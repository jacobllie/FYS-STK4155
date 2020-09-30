from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from X_D import X_D
from sklearn.utils import resample
from functions import beta_,mean_squared_error,R2,FrankeFunction,var
import numpy as np
import matplotlib.pyplot as plt

def OLS(x,y,z,degree,noise,z_star):

    X = X_D(x,y,5)
    beta = beta_(X,np.ravel(z))
                     #We'll use this for the confidence interval of 95 percent
    var_beta = var(X,noise)
    std_beta = np.sqrt(var_beta)*z_star




    #OlS without scaled data for fith degree polynomial
    """
    beta = beta_(X,z)

    z_tilde = X.dot(beta)
    z_tilde_plot = np.reshape(z_tilde,(n,n))
    MSE = mean_squared_error(z,z_tilde)

    print("Degree = 5")
    print("MSE = %.3f"%MSE)
    R2_score = R2(z,z_tilde,np.mean(z))
    print("R2 score = %.3f"%R2_score)
    """

    #finding confidence interval
    #We have stochastic noise with sigma = 1
    #We can substract population mean from sample mean and divide by 1/sqrt(n) the standard deviation
    """
    fig = plt.figure()
    ax = fig.gca(projection="3d")

    # Plot the surface.
    surf = ax.plot_surface(x, y, z_tilde_plot, cmap=cm.coolwarm,linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()"""


    #Scaling the data, to limit the most extreme points

    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    scaler = StandardScaler()
    deg = np.linspace(1,degree,degree)
    MSE_train = np.zeros(degree)
    MSE_test = np.zeros(degree)
    R2_score = np.zeros(degree)

    x_train,x_test,y_train,y_test,z_train,z_test = train_test_split(x,y,z,test_size = 0.3)

    z_train = np.ravel(z_train)
    z_test = np.ravel(z_test)


    MSE_minimum = 10      #The highest MSE we'll allow
    for i in range(1,degree+1):
        print("Degree = %.3f"%i)
        X_train = X_D(x_train,y_train,i)               #X will have the same values on the columns
        X_test = X_D(x_test,y_test,i)
        scaler.fit(X_train)

        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_train_scaled[:,0] = 1                   #setting the first column = 1 because standard scaler sets it to 0
        X_test_scaled[:,0] = 1

        beta_scaled =  beta_(X_train_scaled,z_train)
        z_tilde_scaled_train = X_train_scaled.dot(beta_scaled)
        z_tilde_scaled_test = X_test_scaled.dot(beta_scaled)

        MSE_train[i-1] = mean_squared_error(z_train,z_tilde_scaled_train)
        MSE_test[i-1] = mean_squared_error(z_test,z_tilde_scaled_test)
        R2_score[i-1] = R2(z_train,z_tilde_scaled_train,np.mean(z_train))

        if MSE_test[i-1] < MSE_minimum:
            ztilde_best = np.reshape(z_tilde_scaled_test,(x_test.shape[0],x_test.shape[1]))
            MSE_minimum = MSE_test[i-1]

    return MSE_train,MSE_test,beta,std_beta,x_test,y_test,ztilde_best

if __name__ == '__main__':
    np.random.seed(102)
    n = 15
    z_star = 1.96 #We want 95% confidence inerval
    x = np.random.uniform(0,1,n)
    y = np.random.uniform(0,1,n)
    x = np.sort(x)
    y = np.sort(y)
    x,y = np.meshgrid(x,y)
    noise = 0.01
    noise_arr = noise*np.random.randn(n,n)
    z =FrankeFunction(x,y)+noise
    degree = 20
    deg = np.linspace(0,degree,degree)
    MSE_train,MSE_test,beta,std_beta,_,_,_ = OLS(x,y,z,degree,noise,z_star)

    plt.plot(deg,MSE_train,label="Train")
    plt.plot(deg,MSE_test,label="Test")
    plt.legend()
    plt.show()

    x_axis = np.linspace(0,len(beta),len(beta))
    plt.title("Beta coefficients with their confidence intervals")
    plt.errorbar(x_axis,beta,std_beta,fmt="o")
    plt.show()
