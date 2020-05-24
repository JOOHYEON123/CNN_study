import numpy as np

def step(x):            #step function
    y = x > 0
    return y.astype(np.int)

def sigmoid(x):         #sigmoid function
    return 1/(1 + np.exp(-x))

def ReLU(x):            #ReLU function
    return np.maximum(0, x)

def identity(a):        #Identity function
    return a

def softmax(a):         #softmax function
    c = np.max(a)       #overflow 방지 대책
    exp_a = np.exp(a-c)
    exp_a_sum = np.sum(exp_a)
    y = exp_a / exp_a_sum

    return y