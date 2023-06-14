import numpy as np
import time
import heapq

def euclidean_distance(a, b):
    """
    Compute Euclidean distance between two vectors a and b
    """
    return sum((e1-e2)**2 for e1, e2 in zip(a,b))**0.5