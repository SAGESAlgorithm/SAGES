import numpy as np

def softmax(z, temp):
    z = z / temp
    return np.exp(z) / sum(np.exp(z))