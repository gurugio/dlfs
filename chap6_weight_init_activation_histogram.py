import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

node_num = 100
hidden_layer_size = 5
activations = {}

for i in range(hidden_layer_size):
    if i == 0: x = np.ramdom.randn(1000, 100)
    else: x = activations[i - 1]

    # std is 1
    w = np.random.randn(node_num, node_num) * 1
    a = np.dot(x, w)
    z = sigmoid(a)
    activations[i] = z

for i, a in activations.items():
    
