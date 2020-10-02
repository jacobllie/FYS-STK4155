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
terrain = terrain_data("Calgary.tif")
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
#scaler = StandardScaler()
#scaler.fit(terrain)
#terrain_scaled = scaler.transform(terrain)
terrain_scaled = (terrain-np.min(terrain))/(np.max(terrain)-np.min(terrain))



#First we're doing the OLS

#contour_plot(x,y,np.reshape(terrain_scaled,(n,n)))
_,MSE_OLS,beta_best,_,x_test,y_test,ztilde,i_best = OLS(x,y,np.reshape(terrain_scaled,(n,n)),degree,0,0)
print("OLS")
print("n = {:} degree = {:}".format(n,deg[np.argmin(MSE_OLS)]))
print("MSE = {:.4}".format(np.min(MSE_OLS)))
print(ztilde.shape)
"""contour_plot(x_test,y_test,ztilde)


fig = plt.figure()
ax = fig.gca(projection="3d")

surf = ax.plot_surface(x_test, y_test, ztilde, cmap=cm.coolwarm,linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()"""



terrain = np.reshape(terrain_scaled,(n,n))
x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
X,Y = np.meshgrid(x,y)
print(beta_best.shape)
z_data = X_D(X,Y,i_best).dot(beta_best).reshape(100,100)

plt.subplot(1,2,1)
plt.imshow(np.reshape(terrain_scaled,(n,n)), cmap="gist_earth")
plt.subplot(1,2,2)
plt.imshow(z_data, cmap="gist_earth")
plt.show()


"""
plt.title("Mean squared error with OLS on terrain data")
plt.xlabel("Complexity")
plt.ylabel("MSE")
plt.plot(deg,MSE_OLS,label="OLS")
plt.legend()
plt.show()
"""
"""
#Then we're doing OLS with bootstrap
B = 10
method = "ols"
degree = 30
_,MSE_boot,_,_,_ = bootstrap(B,x,y,np.reshape(terrain_scaled,(n,n)),method,0,degree)
print("OLS with bootstrap")
print("n = {:} degree = {:} bootstraps = {:}".format(n,deg[np.argmin(MSE_boot)],B))
print("MSE = {:.4}".format(np.min(MSE_boot)))
plt.title("Mean squared error with bootstrap & OLS on terrain data")
plt.xlabel("Complexity")
plt.ylabel("MSE")
plt.plot(deg,MSE_boot,label="Bootstrap OLS")
plt.legend()
plt.show()
"""

"""
#Then we're doing OLS with k fold cross validation
k = 5
degree = 30
MSE_cross,_,_,_ = cross_validation(k,x,y,terrain_scaled,degree,"ols",0)
print("OLS with k fold cross validation")
print("n = {:} degree = {:} k = {:}".format(n,deg[np.argmin(MSE_cross)],k))
print("MSE = {:.4}".format(np.min(MSE_cross)))
plt.plot(deg,MSE_cross,label="Cross OLS")
plt.xlabel("Complexity")
plt.ylabel("MSE")
plt.legend()
plt.show()
"""

#Then we're doing Ridge with bootstrap and cross validation


"""
k = 5
B = 10
degree =20
method = "ridge"
lambda_ = np.logspace(-4,1,5)
min_error_boot,min_error_cross,degree_index_boot,degree_index_cross,ridge_heatmap_boot,ridge_heatmap_cross = ridge(x,y,terrain_scaled,k,B,lambda_,degree)
print("----------Bootstrap----------")
print("The best error with %.2f bootstraps and lambda %.5f and degree = %.2f"%(B,lambda_[np.argmin(min_error_boot)],degree_index_boot[np.argmin(min_error_boot)]))
print(np.min(min_error_boot))
print("----------Cross validation---------")
print("The best error with %.2f folds and lambda %.5f and degree = %.2f"%(k,lambda_[np.argmin(min_error_cross)],degree_index_cross[np.argmin(min_error_cross)]))
print(np.min(min_error_cross))
diff = np.min(min_error_boot)/np.min(min_error_cross)
print("Cross validation was {:.3} better".format(diff))


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
"""
lambda_ = np.logspace(-4,0,5)
B = 100
k = 5
min_error_boot,min_error_cross,lasso_heatmap_boot,lasso_heatmap_cross,degree_index_boot,degree_index_cross = lasso(lambda_,degree,x,y,terrain_scaled,k,B)
print("---------Lasso---------")
print("----------Bootstrap----------")
print("The best error with %.2f bootstraps and lambda %.5f and degree = %.2f"%(B,lambda_[np.argmin(min_error_boot)],degree_index_boot[np.argmin(min_error_boot)]))
print(np.min(min_error_boot))
print("----------Cross validation---------")
print("The best error with %.2f folds and lambda %.5f and degree = %.2f"%(k,lambda_[np.argmin(min_error_cross)],degree_index_cross[np.argmin(min_error_cross)]))
print(np.min(min_error_cross))
diff = np.min(min_error_boot)/np.min(min_error_cross)
print("Cross validation was {:.3} better".format(diff))


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
