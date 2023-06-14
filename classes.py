import numpy as np
import time
import heapq

def euclidean_distance(a, b):
    """
    Compute Euclidean distance between two vectors a and b
    """
    return sum((e1-e2)**2 for e1, e2 in zip(a,b))**0.5

def normalize_data(data):
    """
    Normalize data to have zero mean and unit variance
    """
    mean = np.mean(data[:, 1:], axis=0)
    std = np.std(data[:, 1:], axis=0)
    std[std == 0] = 1  # To prevent division by zero
    data[:, 1:] = (data[:, 1:] - mean) / std
    return data