import sys, os

import matplotlib.pyplot as plt
import numpy as np
from chap3_minist import load_mnist
from PIL import Image


def step_function(x):
    # y = x > 0
    # return y.astype(np.int)
    return np.array(x > 0, dtype=np.int)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def identity_function(x):
    return x

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)

def cross_entropy_error(y, t):
    # delta = 1e-7
    # return -np.sum(t * np.log(y + delta))
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size

def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2 * h)

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val
    return grad

def gradient_descent(f, init_x, lr = 0.01, step_num = 100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x

def function_2(x):
    return x[0] ** 2 + x[1] ** 2

if __name__ == '__main__':
    # (x_train, t_train), (x_test, t_test) = \
    #     load_mnist(normalize = True, one_hot_label = True)

    # print(x_train.shape)
    # print(t_train.shape)

    # train_size = x_train.shape[0]
    # batch_size = 10
    # batch_mask = np.random.choice(train_size, batch_size)
    # print(batch_mask)
    # x_batch = x_train[batch_mask]
    # print(x_batch.shape)
    # t_batch = t_train[batch_mask]
    # print(t_batch.shape)

    init_x = np.array([-3.0, 4.0])
    print(gradient_descent(function_2, init_x = init_x, lr = 0.01, step_num = 1000))
    
