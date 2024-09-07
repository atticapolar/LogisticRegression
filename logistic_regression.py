import numpy as np
import data
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

class SGDLogisticRegressor:
    def __init__(self, lr, num_iter):
        self.weights = None
        self.bias = None
        self.lr = lr
        self.num_iter=num_iter