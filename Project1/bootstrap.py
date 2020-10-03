import numpy as np
from numpy.random import randint, randn
from time import time
import matplotlib.pyplot as plt
from scipy.stats import norm

# Returns mean of bootstrap samples
def stat(data):
    return np.mean(data)

# Bootstrap algorithm
def bootstrap(data, statistic, R):
    t = np.zeros(R); n = len(data); inds = np.arange(n); t0 = time()
    # non-parametric bootstrap
    for i in range(R):
        t[i] = statistic(data[randint(0,n,n)])
        # analysis
        print("Runtime:%gsec" % (time()-t0)); print("Bootstrap Statistics :")
        print("original    bias        std.              error")
        print("%8g %10g %14g %15g" % (statistic(data), np.std(data), np.mean(t), np.std(t)))
        return t

mu, sigma = 100, 15
datapoints = 10000
x = mu + sigma*randn(datapoints)
# bootstrap returns the data sample
t = bootstrap(x, stat, datapoints)
# the histogram of the bootstrapped  data
n, binsboot, patches = plt.hist(t, 50, facecolor="red", alpha=0.75)
# add a ’best fit’ line
y = norm.pdf(binsboot, np.mean(t), np.std(t))
lt = plt.plot(binsboot, y, "r--", linewidth=1)
plt.xlabel("Smarts")
plt.ylabel("Probability")
plt.axis([99.5, 100.6, 0, 3.0])

plt.grid(True)
plt.show()
