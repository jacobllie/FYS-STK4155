# Project 1 FYS-STK4155

## Regression analysis and resampling methods

In this project, we explore various regression methods. First, we consider a data set with added noise generated from the Franke function.
Then we apply the same methods to predict contours from real terrain data.

To benchmark the code, we find it most useful to test task g, in the script g.py. This produces contour plots and images from the terrain data. To speed up computations, try running up to a lower degree, e.g. degree = 20 on line 20. You can also try reducing data point, e.g from 50 to 30 (this speeds up computations a lot). These changes to the program already exists in g_test.py in the the test folder of this repository. Simply download the file and run

\> python3 g_test.py

The program will output a lot of warnings, which comes from Sci-kit learn's modules. However, it should be fine as long as you don't encounter any errors.
