# Source Code

*Add documentation*

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



Analyse fruit data set with Keras convolutional neural network (faster and independent of resolution. Be aware that if resolution is too high, it may cause the memory to be fully allocated and cause the program to crash).

```bash
python3 CNN_run.py
```

You will get a set of options when running the script:

- Gray-scale image [Y/n]
- Quadratic resolution of data (N)
- If two of the parameters \eta, \lambda, epochs and batch size should be all single valued or be a set of values [single/set] (use set of values if you want to e.g. create an accuracy map)


