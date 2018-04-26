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

    def error(self, test, size):
        wrong = 0
        """ predict the label of each point in the training dataset using that training dataset"""
        for i in range(size):
            if self.predict(self.train, test[i]) == False:
                wrong += 1
        accuracy = wrong/size
        return accuracy

    def predicttrain(self):
        print("Training Error:" + str(self.error(self.train, 1999)))
    
    def predictvalid(self):
        print("Validation Error:"+ str(self.error(self.valid, 999)))
    
    def predicttest(self):
        print("Test Error: " + str(self.error(self.test, 999)))

    def predict(self, a, point):
        neighbors = dict()

        """ calculate the euclidean distance from the point and map it as a key to label 
            inside the neighbors dictionary
        """
        for i in range(len(a)-1):
            e_dist = np.linalg.norm(a[i][:784] - point[:784])
            neighbors[e_dist] = a[i][784]
        
        """ sort the keys/distances from the point """
        sorted_neighbors = sorted(neighbors.keys())
        first_k = list()

        """ add the first k labels taken from the sorted keys """
        for i in range(self.k):
            first_k.append(neighbors[sorted_neighbors[i]])
        
        """take the mode from the first k closest points"""
        predicted_label = max(set(first_k), key = first_k.count)

        if predicted_label == point[784]:
            return True
        return False

