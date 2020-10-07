import numpy as np
import matplotlib.pyplot as plt
from functions import FrankeFunction,mean_squared_error,X_D
import sklearn.linear_model as skl
import seaborn as sb
from sklearn.preprocessing import StandardScaler
from b import bootstrap
from c import cross_validation



def lasso(lambda_,degree,x,y,z,k,B):
    MSE_lasso_boot = np.zeros(degree)
    MSE_lasso_cross = np.zeros(degree)
    min_error_boot = np.zeros(len(lambda_))
    min_error_cross = np.zeros(len(lambda_))
    lasso_heatmap_boot = np.zeros((len(lambda_),degree))
    lasso_heatmap_cross = np.zeros((len(lambda_),degree))
    degree_index_boot = np.zeros(len(lambda_))
    degree_index_cross = np.zeros(len(lambda_))
    deg = np.linspace(1,degree,degree)
    i = 0
    for lambdas in lambda_:
        _,MSE_lasso_boot,bias,variance,min_error_boot[i] = bootstrap(B,x,y,z,"lasso",lambdas,degree)
        lasso_heatmap_boot[i] = MSE_lasso_boot
        degree_index_boot[i] = deg[np.argmin(MSE_lasso_boot)]
        MSE_lasso_cross,_,_,min_error_cross[i],_,_,_ = cross_validation(k,x,y,z,degree,"lasso",lambdas)
        lasso_heatmap_cross[i] = MSE_lasso_cross
        degree_index_cross[i] = deg[np.argmin(MSE_lasso_boot)]
        #plt.plot(deg,MSE_lasso_boot,label="error")
        #plt.plot(deg,bias,label="bias")
        #plt.plot(deg,variance,label="variance")
        i+=1
    #sb.color_palette("viridis_r", as_cmap=True)
        """
        plt.title("Bootstrap vs Cross for B = {:d} k = {:d} $\lambda$ = {:.4} n = {:d}".format(B,k,lambdas,n))
        plt.plot(deg,MSE_lasso_boot,label="boot")
        plt.plot(deg,MSE_lasso_cross,label="cross")
        plt.legend()
        plt.savefig("./figures/lasso_boot_vs_cross{:d}.pdf".format(i),bbox_inches = 'tight',pad_inches = 0.1,dpi=1200)
        plt.show()"""
    """
    heatmap = sb.heatmap(lasso_heatmap_boot,annot=True,cmap="viridis_r",yticklabels=lambda_,cbar_kws={'label': 'Mean squared error'})
    heatmap.set_xlabel("Complexity")
    heatmap.set_ylabel("$\lambda$")
    heatmap.invert_yaxis()
    heatmap.set_title("Heatmap made from {:} bootstraps".format(B))
    plt.show()

    heatmap = sb.heatmap(lasso_heatmap_cross,annot=True,cmap="viridis_r",yticklabels=lambda_,cbar_kws={'label': 'Mean squared error'})
    heatmap.set_xlabel("Complexity")
    heatmap.set_ylabel("$\lambda$")
    heatmap.invert_yaxis()
    heatmap.set_title("Heatmap made from {:} folds in cross validation".format(k))
    plt.show()
    """
    """
    plt.plot(deg,min_error_boot,label="bootstrap")
    plt.plot(deg,min_error_cross,label="cross validation")
    plt.legend()
    plt.show()
    """
    return min_error_boot,min_error_cross,lasso_heatmap_boot,lasso_heatmap_cross,degree_index_boot,degree_index_cross
if __name__ == '__main__':
    degree = 20
    lambda_ = np.array([1e-4,1e-3,1e-2,1e-1,1])
    np.random.seed(130)
    n = 50
    x = np.random.uniform(0,1,n)
    y = np.random.uniform(0,1,n)
    noise = 0.5 * np.random.randn(n,n)
    x = np.sort(x)
    y = np.sort(y)
    x,y = np.meshgrid(x,y)
    z = np.ravel(FrankeFunction(x,y)+noise)
    B = 75
    bias = np.zeros(degree)
    variance = np.zeros(degree)
    k = 5
    min_error_boot,min_error_cross,lasso_heatmap_boot,lasso_heatmap_cross,degree_index_boot,degree_index_cross = lasso(lambda_,degree,x,y,z,k,B)
    print("---------Lasso---------")
    print("----------Bootstrap----------")
    print("The best error with %.2f bootstraps and lambda %.5f and degree = %.2f"%(B,lambda_[np.argmin(min_error_boot)],degree_index_boot[np.argmin(min_error_boot)]))
    print(np.min(min_error_boot))
    print("----------Cross validation---------")
    print("The best error with %.2f folds and lambda %.5f and degree = %.2f"%(k,lambda_[np.argmin(min_error_cross)],degree_index_cross[np.argmin(min_error_cross)]))
    print(np.min(min_error_cross))
    diff = np.min(min_error_boot)/np.min(min_error_cross)
    print("Cross validation was {:.3} better".format(diff))


    heatmap = sb.heatmap(lasso_heatmap_boot.T,annot=True,cmap="viridis_r",xticklabels=lambda_,cbar_kws={'label': 'Mean squared error'},fmt = ".5")
    heatmap.set_xlabel("$\lambda$")
    heatmap.set_ylabel("Complexity")
    heatmap.invert_yaxis()
    heatmap.set_title("Heatmap made from {:} bootstraps".format(B))
    fig = heatmap.get_figure()
    fig.savefig("./figures/lasso_heatmap_boot.pdf",bbox_inches = 'tight',pad_inches = 0.1,dpi=1200)
    plt.show()

    heatmap = sb.heatmap(lasso_heatmap_cross.T,annot=True,cmap="viridis_r",xticklabels=lambda_,cbar_kws={'label': 'Mean squared error'},fmt=".5")
    heatmap.set_xlabel("$\lambda$")
    heatmap.set_ylabel("Complexity")
    heatmap.invert_yaxis()
    heatmap.set_title("Heatmap made from {:} folds in cross validation".format(k))
    fig = heatmap.get_figure()
    fig.savefig("./figures/lasso_heatmap_cross.pdf",bbox_inches = 'tight',pad_inches = 0.1,dpi=1200)
    plt.show()
