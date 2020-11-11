# Project 2 FYS-STK4155 Autumn 2020

## Classification and Regression, from linear and logistic re-gression to neural networks

In this project, we explore stochastic gradient descent and feedforward neural networks with back-propagation.

First off, we wish to thank Elin Finstad and Anders Bråte for collaboration and great discussions. We also wish to thank the FYS-STK4155 team for all help.

To run the programs we have made, download our repository and make sure you are in the src folder.

```bash
cd src
```

Now, you can run the scripts. We recommend starting off with analysing stochastic gradient descent on the Franke data set.
You can do this by running the regression script:

```bash
python3 regression.py
```

Now your terminal will hopefully give you the options to analyse minibatches and epochs. Respond with Y for yes or n for no.
The other option to analyse is learning schedule. Both analyses produce a heatmap and saves them in the folder ```figures```. This program takes a few minutes to run. If you don't have 5 minutes, we recommend running the NN script. We have included a progress bar so you can track the progress.

The next script we recommend to analyse is the neural network (NN), which analyses the Franke function when run. Run the following command

```bash
python3 NN.py
```

Now you will get following options to analyse

- learning rate vs epochs

- learning rate vs penalty parameter

- Keras (performs analysis of learning rate and penalty parameter using Keras library)

- relu activation (learning rate vs epochs) (changes activation function in hidden layers)

Note that the analysis with Keras takes some time. The quickest analysis to perform is the first option, learning rate vs epochs. This is the analysis we recommend performing to test our code. All analyses produces heatmaps and saves them in the folder ```figures```, and all analyses comes with a progress bar to track progress.

The last script we have made easy to analyse is ```digit.py```. This is the script which trains the network on the handwritten digits from MNIST. Run the program with

```bash
python3 digit.py
```
The program will by default produce a confusion matrix which shows how the network predicts digits. Optionally, you can analyse how learning rate and penalty
penalty parameter changes the networks' performance. Simply press y and hit enter when the terminal asks you

```Analyse penalty parameter and learning rate [Y/n]: ```

Keep in mind that different random seeds in the neural network affects results.

The last program we have is ```log-reg.py```, which performs logistic regression. This program uses logistic regression to analyse digits from MNIST. It produces a heatmap of accuracy from both scikit learn and our own neural network with no hidden layers. It also produces a confusion matrix from our network. To run this program:

```bash
python3 log-reg.py
```

All other scripts in the folder ```src``` contain support functions.
