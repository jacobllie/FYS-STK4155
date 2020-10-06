import matplotlib.pyplot as plt
import numpy as np
from functions import FrankeFunction,beta_,mean_squared_error,beta_r,R2,terrain_data,contour_plot,X_D
from a import OLS
from b import bootstrap
from c import cross_validation
from d import ridge
from e import lasso
import seaborn as sb
from matplotlib.pyplot import cm
from sklearn.preprocessing import StandardScaler
terrain = terrain_data("Kamloops.tif")
n = 50
terrain = terrain[:n,:n]      #this is the z
n = len(terrain)
print(terrain.shape)
x = np.linspace(0,1,len(terrain))
y = np.linspace(0,1,len(terrain))
x,y = np.meshgrid(x,y)
degree = 20
deg  = np.linspace(1,degree,degree)
terrain = np.ravel(terrain)
terrain_scaled = (terrain-np.min(terrain))/(np.max(terrain)-np.min(terrain))


file = open("g_result.txt","w")


#First we're doing the OLS

_,MSE_OLS,_,_,ztilde,i_best,beta_best = OLS(x,y,np.reshape(terrain_scaled,(n,n)),degree,0,0)
file.write("OLS\n")
file.write("n = {:} degree = {:}\n".format(n,deg[np.argmin(MSE_OLS)]))
file.write("MSE = {:.4}\n".format(np.min(MSE_OLS)))

k=5
_,_,_,_, beta_best, i_best, _ = cross_validation(k,x,y,terrain_scaled,degree,"ols",0)


terrain = np.reshape(terrain_scaled,(n,n))

x_mrk = np.linspace(0, 1, 100)
y_mrk = np.linspace(0, 1, 100)
X,Y = np.meshgrid(x_mrk,y_mrk)
z_data = X_D(X,Y,i_best).dot(beta_best).reshape(100,100)

plt.subplot(1,2,1)
plt.title("Terrain data")
plt.imshow(np.reshape(terrain_scaled,(n,n)), cmap="gist_earth")
plt.subplot(1,2,2)
plt.title("Model")
plt.imshow(z_data, cmap="gist_earth")
plt.savefig("figures/terrain_image.pdf")
plt.show()

plt.subplot(1,2,1)
plt.title("Terrain data")
plt.contour(x,y,np.reshape(terrain_scaled,(n,n)))
plt.subplot(1,2,2)
plt.title("Model")
cp = plt.contour(X,Y,z_data)
plt.colorbar(cp) # Add a colorbar to a plot
plt.savefig("figures/terrain_contour.pdf")
plt.show()



plt.title("Mean squared error with OLS on terrain data")
plt.xlabel("Complexity")
plt.ylabel("MSE")
plt.plot(deg,MSE_OLS,label="OLS")
plt.legend()
plt.show()




#Then we're doing OLS with bootstrap
B = 75
method = "ols"
_,MSE_boot,_,_,_ = bootstrap(B,x,y,terrain_scaled,method,0,degree)

file.write("OLS with bootstrap\n")
file.write("n = {:} degree = {:} bootstraps = {:}\n".format(n,deg[np.argmin(MSE_boot)],B))
file.write("MSE = {:.4}\n".format(np.min(MSE_boot)))

"""
plt.title("Mean squared error with bootstrap & OLS on terrain data")
plt.xlabel("Complexity")
plt.ylabel("MSE")
plt.plot(deg,MSE_boot,label="Bootstrap OLS")
plt.legend()
plt.show()
"""

"""
#Then we're doing OLS with k fold cross validation
MSE_cross,_,_,_ = cross_validation(k,x,y,terrain_scaled,degree,"ols",0)

file.write("OLS with k fold cross validation\n")
file.write("n = {:} degree = {:} k = {:}\n".format(n,deg[np.argmin(MSE_cross)],k))
file.write("MSE = {:.4}\n".format(np.min(MSE_cross)))
"""


"""
plt.plot(deg,MSE_cross,label="Cross OLS")
plt.xlabel("Complexity")
plt.ylabel("MSE")
plt.legend()
plt.show()
"""

#Then we're doing Ridge with bootstrap and cross validation





method = "ridge"
lambda_ = np.logspace(-4,0,5)
min_error_boot,min_error_cross,degree_index_boot,degree_index_cross,ridge_heatmap_boot,ridge_heatmap_cross = ridge(x,y,terrain_scaled,k,B,lambda_,degree)


file.write("Ridge with Bootstrap\n")
file.write("The best error with %.2f bootstraps and lambda %.5f and degree = %.2f\n"%(B,lambda_[np.argmin(min_error_boot)],degree_index_boot[np.argmin(min_error_boot)]))
file.write("MSE = {:.4}".format(np.min(min_error_boot)))
file.write("\n")
file.write("Ridge with Cross Validation\n")
file.write("The best error with %.2f folds and lambda %.5f and degree = %.2f\n"%(k,lambda_[np.argmin(min_error_cross)],degree_index_cross[np.argmin(min_error_cross)]))
file.write("MSE = {:.4}".format(np.min(min_error_cross)))
file.write("\n")
diff = np.min(min_error_boot)/np.min(min_error_cross)
file.write("min min_error_boot/min_error_cross = {:.4}\n".format(diff))


"""
heatmap = sb.heatmap(ridge_heatmap_boot,annot=True,cmap="viridis_r",yticklabels=lambda_,cbar_kws={'label': 'Mean squared error'})
heatmap.set_xlabel("Complexity")
heatmap.set_ylabel("$\lambda$")
heatmap.invert_yaxis()
heatmap.set_title("Heatmap made from {:} bootstraps".format(B))
plt.show()

heatmap = sb.heatmap(ridge_heatmap_cross,annot=True,cmap="viridis_r",yticklabels=lambda_,cbar_kws={'label': 'Mean squared error'})
heatmap.set_xlabel("Complexity")
heatmap.set_ylabel("$\lambda$")
heatmap.invert_yaxis()
heatmap.set_title("Heatmap made from {:} folds in cross validation".format(k))
plt.show()
"""

#Lasso with cross validation and bootstrap

lambda_ = np.logspace(-4,0,5)

min_error_boot,min_error_cross,lasso_heatmap_boot,lasso_heatmap_cross,degree_index_boot,degree_index_cross = lasso(lambda_,degree,x,y,terrain_scaled,k,B)
file.write("Lasso with Bootstrap\n")
file.write("The best error with %.2f bootstraps and lambda %.5f and degree = %.2f\n"%(B,lambda_[np.argmin(min_error_boot)],degree_index_boot[np.argmin(min_error_boot)]))
file.write("MSE = {:.4}\n".format(np.min(min_error_boot)))
file.write("Lasso with Cross Validation\n")
file.write("The best error with %.2f folds and lambda %.5f and degree = %.2f\n"%(k,lambda_[np.argmin(min_error_cross)],degree_index_cross[np.argmin(min_error_cross)]))
file.write("MSE = {:.4}\n".format(np.min(min_error_cross)))
diff = np.min(min_error_boot)/np.min(min_error_cross)
file.write("min_error_boot/min_error_cross = {:.4}".format(diff))


file.close()

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
