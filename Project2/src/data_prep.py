import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class data_prep:
    def __init__(self):
        self.D = None

    def MNIST(self,data,inputs,labels):
        # ensure the same random numbers appear every time
        self.z = labels
        np.random.seed(0)

        print("inputs = (n_inputs, pixel_width, pixel_height) = " + str(inputs.shape))
        print("labels = (n_inputs) = " + str(labels.shape))


        # flatten the image
        # the value -1 means dimension is inferred from the remaining dimensions: 8x8 = 64
        n_inputs = len(inputs)
        inputs = inputs.reshape(n_inputs, -1)    #making feature matrix
        print("X = (n_inputs, n_features) = " + str(inputs.shape))
        self.D = inputs


    def X_D(self, x, y, z, p):
        self.z = z
        #if the x, and y values are matrices we need to make them into vectors
        if len(x.shape)>1:
            x = np.ravel(x)    #flatten x
            y = np.ravel(y)
        self.N = len(x)
        l = int((p+1)*(p+2)/2) #number of columns in beta

        X = np.ones((self.N,l))

        for i in range(1,p+1):     #looping over the degrees but we have 21 elements
            q = int(i*(i+1)/2)           #getting the odd numbers 1.3.5 etc.
            for k in range(i+1):    #this loop makes sure we dont fill the same values in X over and over
                X[:,q+k] = x**(i-k)*y**k   #using the binomial theorem to get the right exponents
        self.D = X

    def train_test_split_scale(self):
        X_train, X_test, z_train, z_test = train_test_split(self.D, self.z, test_size=0.3)
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_scaled =  scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        if np.all(self.D[:,0] == 1):
            X_train_scaled[:,0] = 1
            X_test_scaled[:,0] = 1
        return X_train_scaled, X_test_scaled, z_train, z_test
