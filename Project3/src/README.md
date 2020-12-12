# Source Code

*Documentation*

## FFNN
Analyse fruit data set with our FFNN neural network (slow and not recommended):

```bash
python3 fruit_NN.py

Color or Black-white [C/Bw]: 
Size of N images (N x N pixels): 
Save network? [Y/n]: 
```

Choose color (C) or gray-scale (Bw), then choose picture resolution (go for something low for quick calculations, e.g. 10). Not neccessary to save network (n)
 
Analyse fruit data set with Keras neural network (less slow, couple minutes with 50x50 gray-scale):
 
```bash
python3 fruit_keras.py

Color or Black-white [C/Bw]: 
Size of N images (N x N pixels): 
```

Choose color (C) or gray-scale (Bw), then choose picture resolution (50 pixels gray-scale takes around 20 min).

 ## CNN

Running the Convolutional Neural Network class CNN_class.py, you can either run CNN1.py or CNN_run.py.
They are very similar, but the difference lays in how many hyperparameter you can change. 

| Parameter            | CNN1  | CNN_run |
|----------------------|-------|---------|
| lmbd                 | <ul><li>- [x] </li> | <ul><li>- [x] </li>   |
| eta                   | <ul><li>- [x] </li> | <ul><li>- [x] </li>   |
| epochs               | x     | <ul><li>- [x] </li>   |
| batch size           | x     | <ul><li>- [x] </li>   |
| number of kernels    | x     | <ul><li>- [x] </li>   |
| neurons hidden layer | x     | <ul><li>- [x] </li>   |


Analyse fruit data set with Keras convolutional neural network (faster and independent of resolution. Be aware that if resolution is too high, it may cause the memory to be fully allocated and cause the program to crash).

```bash
python3 CNN_run.py
```

You will get a set of options when running the script:

- Gray-scale image [Y/n]
- Quadratic resolution of data (N)
- If two of the parameters \eta, \lambda, epochs and batch size should be all single valued or be a set of values [single/set] (use set of values if you want to e.g. create an accuracy map)

Running CNN1.py the maximum number of images that a training run can take is default 3000 images. Be sure that your RAM can handle this amount. 
Running the program for colors, optimal learning rates and penalties and 100x100 resolution image will take approximately 5 minutes.

```bash
python3 CNN1.py

gray/color?: 
You can either run for optimal learning rate and penalty. 
      Or you can run for different learning rates and penalty
Write [optimal/different]
Save Confusion matrix [Y/n] 
Save validation accuracy [Y/n] 
Save training accuracy for epochs [Y/n] 
How big resolution? NxN: 

```
Note that the program does not plot anything, so remember to save the results you want to plot (CNN_plot.py is the program to run if you want to plot results).

**For a faster run choose gray, optimal, y for the values you wish to store and a low resolution (e.g. 10 pixels).**

## Plotting

