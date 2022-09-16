
# neccessory libraries
from re import T
from this import d
import panda as pd
import numpy as np
from matplotlib import pyplot as plt

data = pd.read_read('/kaggle/input/digit-recognizer/train.csv')

# displaythe table
data.head

# to convert to arrays
data.np.array(data)
m, n = data.shape  # n=784+1

data_dev = data[0:1000].T  # for checking accuracy
y_dev = data_dev[0]
x_dev = data_dev[1:n]


# for training data
data_train = data[1000:m].T

y_train = data_train[0]
x_train = data_train[1:n]

x_train[:, 0].shape
y_train

# setting random values to the arrays


def inital_parameter(W1, B1, W2, B2):
    W1 = np.random.rand(10, 784) - 0.5
    B1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    B2 = np.random.rand(10, 1) - 0.5
    return W1, B1, W2, B2


def ReLu(Z):
    # return X if X > 0
    # or return 0
    # Linear Line
    return np.maximum(0, Z)


def soft_max(Z):
    # e^(Z)/sum of e^(Z*j) for j:1,i
    # returns probability 0.00 to 1.00
    return np.exp(Z)/np.sum(np.exp(Z))


def forward_prop(W1, B1, W2, B2, X):
    Z1 = W1.dot[X] + B1
    A1 = ReLu(Z1)
    Z2 = W2.dot[A1] + B2
    A2 = soft_max(Z2)
    return Z1, A1, Z2, A2


def one_hot(Y):
    # for cost finding
    one_hot_Y = np.zeros(Y.size, Y.max()+1)
    one_hot_Y[np.arange(Y.size, Y)] = 1
    one_hot_Y = one_hot_Y.T  # each column an example


def ReLu_dev(Z):
    # return 1 || 0
    return Z > 0


def back_prop(Z1, A1, Z2, A2, W2, X, Y):
    m = Y.size

    # calculating first cost of train

    # 1st hidden layer
    # difference Z = our predistion A1 - data label
    dZ2 = A2 - one_hot(Y)
    # difference of weight due to dZ2 = A1 * dZ2 / m
    dW2 = 1/m * dZ2.dot(A1.T)
    # difference of bias due to dZ2 =  sum of dZ2 / m
    dB2 = 1 / m * np.sum(dZ2, 2)

    # 0th layer
    dZ1 = W2.T.dot(dZ2) * ReLu_dev(Z1)

    dW1 = 1 / m * dZ1.dot(X.T)

    dB1 = 1 / m * np.sum(dZ1, 2)

    # ready for the matrix updation
    return dW1, dB1, dW2, dB2


def update_parameters(W1, B1, W2, B2, dW1, dB1, dW2, dB2, H):
    # updating the matrix
    W1 -= (H*dW1)
    B1 -= (H*dB1)
    W2 -= (H*dW2)
    B2 -= (H*dB2)
    return W1, B1, W2, B2


def return_prediction(A2):
    return np.argmax(A2, 0)


def return_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y)


def start_training(X, Y, iterations, H):
    # random values to the model
    W1, B1, W2, B2 = inital_parameter(W1, B1, W2, B2)

    for i in range(iterations):
        # starting the model
        Z1, A1, Z2, A2 = forward_prop(W1, B1, W2, B2, X)

        # training and finding the cost
        dW1, dB1, dW2, dB2 = back_prop(Z1, A1, Z2, A2, W2, X, Y)

        W1, B1, W2, B2 = update_parameters(
            W1, B1, W2, B2, dW1, dB1, dW2, dB2, H)

        if (i % 50 == 0):
            print("iterations done: ", i)
            print("accuracy is ",  return_accuracy(return_prediction(A2), Y))

    return W1, B1, W2, B2


# start model training and return the model matrix
# it contains the weight and bias trained with ReLu and softmax for printing the possibility of the result 
W1, B1, W2, B2 = start_training(x_train, y_train, 500, 0.1)
