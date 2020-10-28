import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class data_prep:
    def __init__(self,x,y,z,p):
        self.x = x
        self.y = y
        self.z = z
        self.p = p
        self.D = None


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

    def __call__(self):
        self.X_D()
        X_train,X_test,z_train,z_test = train_test_split(self.D,self.z,test_size=0.3)
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_scaled =  scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_train_scaled[:,0] = 1
        X_test_scaled[:,0] = 1
        return X_train_scaled, X_test_scaled, z_train, z_test
