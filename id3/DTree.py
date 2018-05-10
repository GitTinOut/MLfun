import pandas as pd
import math
import numpy as np
from Node import Node

class DTree:
    def __init__(self, train, valid, test):
        self.train = pd.read_csv(train, sep = ' ', header = None).as_matrix()
        self.train = np.delete(self.train, 23, axis = 1)
        self.valid = pd.read_csv(valid, sep = ' ', header = None).as_matrix()
        self.valid = np.delete(self.valid, 23, axis = 1)
        self.test = pd.read_csv(test, sep = ' ', header = None).as_matrix()
        self.test = np.delete(self.test, 23, axis = 1)
        #self.labels = labels
        self.root = None
    def buildTree(self):

        node = self.root
        node = Node(data=self.train.copy())
        impure = [node]
        i = 0
        while(len(impure) != 0):
            
            print(i)
            i += 1
            node = impure.pop(0)
            feat,thres= self.split(node.getData())
            node.setF(feat)
            node.setT(thres)
    
            partition = self.split_data(node.getData(), feat, thres)
            left= partition[0]
            right = partition[1]
        
            l_child = None
            if self.check_purity(left) == True:
                l_child = Node(left, pure = True, label = left[0][22])
            else:
                l_child = Node(left)
                impure.append(l_child)
            
            r_child = None
            if self.check_purity(right) == True:
                r_child = Node(right, pure = True,label = right[0][22])
            else:
                r_child = Node(right)
                impure.append(r_child)
            
            node.setLeft(l_child)
            node.setRight(r_child)

    def test_feat(self, feat):
        node = self.root

        while(node.isPure() ==False):
            node = node.getNext(feat[node.getF()])
        return node.getLabel()
    
    def test_train(self):
        correct = 0
        data = self.train.copy()
        for i in range(data.shape[0]):
            if data[i][22] == self.test_feat(data[i,:22]):
                correct += 1.0
        return correct/self.shape[0]

            
        
    def check_purity(self, data):
        unique = dict()
        print(data[:,22])
        for i in list(data[:,22]):
            unique[i] = 1
        if unique.keys() == 1:
            return True
        else:
            return False

    def split(self, data):
        possible_splits = []
        for i in range(data.shape[1]-2):
            x = self.s_entropy_of_feat(data[:,i], data[:,22])
            possible_splits.append( (x[0],i,x[1]) )
        
        g = sorted(possible_splits)[0]

        return (g[1],g[2])
        

    def s_entropy_of_feat(self, feat, label):
        t = self.thresholds(feat)
        candidates = []
        for i in t:
            candidates.append([self.entropy(feat,label,i),i])
        return sorted(candidates)[0]
    def entropy(self, feat, label,t):
        X = [0,0]
        for i in feat:
            if i < t:
                X[0] += 1
            else:
                X[1] += 1
        X[0] = X[0]/len(feat)
        X[1] = X[1]/len(feat)

        condprob = self.cond_prob(feat,label,t)
        ltt = condprob[0]
        gte = condprob[1]
        for i in range(len(ltt)):
            if ltt[i] == 0.0:
                ltt[i] = 1
        for i in range(len(gte)):
            if gte[i] == 0.0:
                gte[i] = 1
        Hltt = -1* ltt[0]*math.log(ltt[0]) - ltt[1]*math.log(ltt[1])
        Hgte = -1 * gte[0]*math.log(gte[0]) - gte[1]*math.log(gte[1])
        return X[0]*Hltt + X[1]*Hgte

    def thresholds(self, feat):
        unique = dict()
        for i in feat:
           unique[i] = 1

        unique = list(unique.keys())
        thresholds = []
        for i in range( len(unique)-1 ):
            thresholds.append( (unique[i] + unique[i+1])/2. )

        if len(thresholds) == 0:
            thresholds.append(unique[0])
        
        return thresholds

    def cond_prob(self, f, l, t):
        Xlt = [0, 0]
        Xget = [0,0]
        total = 0
        for i in range(len(f)):
            if l[i] == 0.0 and f[i] < t:
                total += 1
                Xlt[0] += 1
            elif l[i] == 1 and f[i] < t:
                total += 1
                Xlt[1] += 1
        if total == 0:
            Xlt[0] = 0
            Xlt[1] = 0
        else:
            Xlt[0] = Xlt[0]/total
            Xlt[1] = Xlt[1]/total
        total = 0
        for i in range(len(f)):
            if l[i] == 0 and f[i] >= t:
                total += 1
                Xget[0] += 1
            elif l[i] == 1 and f[i] >= t:
                total += 1
                Xget[1] += 1
        Xget[0] = Xget[0]/total
        Xget[1] = Xget[1]/total

        return [Xlt, Xget]
    def split_data(self, data, f_index, t):
        l_ind =[]
        r_ind =[]
        for i in range(data.shape[0]):
            if data[i][f_index] < t:
                l_ind.append(i)
            else:
                r_ind.append(i)
        
        left = data[l_ind[0]]
        right = data[r_ind[0]]
        
        first = 0 
        for i in l_ind:
            if first == 0:
                first += 1
                continue
            left = np.vstack([left,data[i]])
        first = 0
        for i in r_ind:
            if first ==0:
                first += 1
                continue
            right = np.vstack([right,data[i]])

        return [left,right]

        


