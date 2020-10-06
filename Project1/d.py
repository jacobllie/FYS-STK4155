
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as skl
from sklearn.metrics import mean_squared_error,r2_score
from functions import beta_,R2,mean_squared_error,beta_r,FrankeFunction
from c import cross_validation
from b import bootstrap
import seaborn as sb

def ridge(x,y,z,k,B,lambda_,degree):
    scaler = StandardScaler()
    deg = np.linspace(1,degree,degree)
    #Its important to send the meshgrid into the design matrix function
    #how many models we'll make   we have int(n/k) values per model
    MSE_ridge_boot = np.zeros(degree)
    MSE_ridge_cross = np.zeros(degree)
    min_error_boot = np.zeros(len(lambda_))
    min_error_cross = np.zeros(len(lambda_))
    ridge_heatmap_boot = np.zeros((len(lambda_),degree))
    ridge_heatmap_cross = np.zeros((len(lambda_),degree))
    degree_index_boot = np.zeros(len(lambda_))
    degree_index_cross = np.zeros(len(lambda_))
    i = 0
    plt.style.use("seaborn")

    for lambdas in lambda_:
        _,MSE_ridge_boot,bias_boot,variance_boot,min_error_boot[i] = bootstrap(B,x,y,z,"ridge",lambdas,degree)
        ridge_heatmap_boot[i] = MSE_ridge_boot
        degree_index_boot[i] = deg[np.argmin(MSE_ridge_boot)]    #the smallest MSE_ridge_boot value will give a degree, which we insert into the degree_index_boot array
        MSE_ridge_cross,_,_,min_error_cross[i],_,_,_ = cross_validation(k,x,y,z,degree,"ridge",lambdas)
        ridge_heatmap_cross[i] = MSE_ridge_cross
        degree_index_cross[i] = deg[np.argmin(MSE_ridge_cross)]
        """plt.title("Bootstrap vs Cross validation $\lambda$ = {:.4} k = {:d} B = {:d} n = {:d}".format(lambdas,k,B,n))
        plt.plot(deg,MSE_ridge_boot,label="error")
        plt.plot(deg,MSE_ridge_cross,label="Cross validation")
        plt.legend()
        plt.savefig("./figures/ridge_boot_vs_cross{:d}.pdf".format(i),bbox_inches = 'tight',pad_inches = 0.1,dpi = 1200)
        plt.show()

        plt.title("Bias Variance $\lambda$ = {:.4} B = {:d} n = {:d} ".format(lambdas,B,n))
        plt.plot(deg,bias_boot,label="bias")
        plt.plot(deg,variance_boot,label="variance")
        plt.plot(deg,MSE_ridge_boot,label="error")
        plt.legend()
        plt.savefig("./figures/ridge_bias_variance{:d}.pdf".format(i),bbox_inches = 'tight',pad_inches = 0.1,dpi = 1200)
        plt.show()
        """
        i += 1

    return min_error_boot,min_error_cross,degree_index_boot,degree_index_cross,ridge_heatmap_boot,ridge_heatmap_cross

if __name__ == '__main__':
    np.random.seed(130)
    n = 50             #The dataset must be divisible with 5
    x = np.random.uniform(0,1,n)
    y = np.random.uniform(0,1,n)
    x = np.sort(x)
    y = np.sort(y)
    x,y = np.meshgrid(x,y)
    noise = 0.1*np.random.randn(n,n)
    z = np.ravel(FrankeFunction(x,y)+noise)

    degree = 30
    deg = np.linspace(1,degree,degree)
    R2_score = np.zeros(degree)
    MSE_ridge_boot = np.zeros(degree)
    MSE_ridge_cross = np.zeros(degree)
    lambda_ = np.array([1e-4,1e-3,1e-2,1e-1,1])


    k = 5
    B = 75

    min_error_boot,min_error_cross,degree_index_boot,degree_index_cross,ridge_heatmap_boot,ridge_heatmap_cross = ridge(x,y,z,k,B,lambda_,degree)
    print("----------Ridge----------")
    print("----------Bootstrap----------")
    print("The best error with %.2f bootstraps and lambda %.5f and degree = %.2f"%(B,lambda_[np.argmin(min_error_boot)],degree_index_boot[np.argmin(min_error_boot)]))
    print(np.min(min_error_boot))
    print("----------Cross validation---------")
    print("The best error with %.2f folds and lambda %.5f and degree = %.2f"%(k,lambda_[np.argmin(min_error_cross)],degree_index_cross[np.argmin(min_error_cross)]))
    print(np.min(min_error_cross))
    diff = np.min(min_error_cross)/np.min(min_error_boot)
    print("Cross validation was {:.3} better".format(diff))


    heatmap = sb.heatmap(ridge_heatmap_boot.T,annot=True,cmap="viridis_r",xticklabels=lambda_,cbar_kws={'label': 'Mean squared error'},fmt=".5")
    heatmap.set_xlabel("$\lambda$")
    heatmap.set_ylabel("Complexity")
    heatmap.invert_yaxis()
    heatmap.set_title("Heatmap made from {:} bootstraps".format(B))
    fig = heatmap.get_figure()
    fig.savefig("./figures/ridge_heatmap_boot.pdf",bbox_inches = 'tight',pad_inches = 0.1,dpi = 1200)
    plt.show()

    heatmap = sb.heatmap(ridge_heatmap_cross.T,annot=True,cmap="viridis_r",xticklabels=lambda_,cbar_kws={'label': 'Mean squared error'},fmt=".5")
    heatmap.set_xlabel("$\lambda$")
    heatmap.set_ylabel("Complexity")
    heatmap.invert_yaxis()
    heatmap.set_title("Heatmap made from {:} folds in cross validation".format(k))
    fig = heatmap.get_figure()
    fig.savefig("./figures/ridge_heatmap_cross.pdf",bbox_inches = 'tight',pad_inches = 0.1,dpi=1200)
    plt.show()
