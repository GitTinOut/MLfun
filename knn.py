import pandas as pd
import numpy as np
from scipy.stats import mode

class KNN:
    def __init__(self, train, valid, test, k):
        self.trainfile = train
        self.validfile = valid
        self.testfile = test
        self.train = pd.read_csv(train,sep = ' ',header = None).as_matrix()
        self.valid = pd.read_csv(valid, sep = ' ',header = None).as_matrix()
        self.test = pd.read_csv(test, sep = ' ', header = None).as_matrix()
        self.k = k

    def setk(self,new):
        self.k = new
    
    """sets the projected training matrix by matrix multiplication"""
    def setp(self, p):
        self.ptrain = np.matmul( np.delete(pd.read_csv(self.trainfile, sep = ' ',header = None).as_matrix(), -1, axis=1),
                                 pd.read_csv(p, sep = ' ', header = None).as_matrix() )
        self.pvalid = np.matmul( np.delete(pd.read_csv(self.validfile, sep = ' ',header = None).as_matrix(), -1, axis=1),
                                 pd.read_csv(p, sep = ' ', header = None).as_matrix() )
        self.ptest = np.matmul( np.delete(pd.read_csv(self.testfile, sep = ' ',header = None).as_matrix(), -1, axis=1),
                                 pd.read_csv(p, sep = ' ', header = None).as_matrix() )

    def error(self, test, orig, size, proj):
        wrong = 0
        if proj == False:
            train = self.train
        else:
            train = self.ptrain

        """ predict the label of each point in the training dataset using that training dataset"""
        for i in range(size):
            if self.predict(train, test[i], proj,i, orig) == False:
                wrong += 1
        accuracy = wrong/size
        return accuracy

    def predicttrain(self, proj = False):
        if proj == False:
            print("Training Error:" + str(self.error(self.train, self.train, 2000, proj)))
        else: 
            print("Training Error:" + str(self.error(self.ptrain, self.train, 2000, proj)))

    
    def predictvalid(self, proj = False):
        if proj == False:
            print("Validation Error:"+ str(self.error(self.valid, self.valid, 1000, proj)))
        else:
            print("Validation Error:"+ str(self.error(self.pvalid, self.valid, 1000, proj)))
    
    def predicttest(self, proj = False):
        if proj == False:
            print("Test Error:" + str(self.error(self.test, self.test, 1000, proj)))
        else:
            print("Test Error:" + str(self.error(self.ptest, self.test, 1000, proj)))

    def predict(self, a, point, proj, index, orig = None):
        neighbors = dict()

        if proj == False:
            features = 784
        else:
            features = 20

        """ calculate the euclidean distance from the point and map it as a key to label 
            inside the neighbors dictionary
        """
        for i in range(len(a)-1):
            e_dist = np.linalg.norm(a[i][:features] - point[:features])
            neighbors[e_dist] = self.train[i][784]
        
        """ sort the keys/distances from the point """
        sorted_neighbors = sorted(neighbors.keys())
        first_k = list()

        """ add the first k labels taken from the sorted keys """
        for i in range(self.k):
            first_k.append(neighbors[sorted_neighbors[i]])
        
        """take the mode from the first k closest points"""
        predicted_label = max(set(first_k), key = first_k.count)
        #print(predicted_label,point[784])
        if predicted_label == orig[index][784]:
            return True
        return False

