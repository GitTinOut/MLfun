import pandas as pd
import numpy as np
from scipy.stats import mode

class KNN:
    def __init__(self, train, valid, test, k):
        self.train = pd.read_csv(train,sep = ' ').as_matrix()
        self.valid = pd.read_csv(valid, sep = ' ').as_matrix()
        self.test = pd.read_csv(test, sep = ' ').as_matrix()
        self.k = k

    def setk(self,new):
        self.k = new

    def predicttrain(self):
        wrong = 0
        for i in range(1999):
            if self.predict(self.train, self.train[i]) == False:
                wrong += 1
        accuracy = wrong/1999
        print(accuracy)

    def predict(self, a, point):
        neighbors = dict()
        for i in range(len(a)-1):
            e_dist = np.linalg.norm(a[i][:784] - point[:784])
            neighbors[e_dist] = a[i][784]

        sorted_neighbors = sorted(neighbors.keys())
        first_k = list()
        for i in range(self.k):
            first_k.append(neighbors[sorted_neighbors[i]])
        predicted_label = max(set(first_k), key = first_k.count)
        print((predicted_label, point[784])) 
        if predicted_label == point[784]:
            return True
        return False

