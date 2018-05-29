import pandas as pd
import numpy as np
class Perceptron:
    def __init__(self, train, test, dictionary):
        d = pd.read_csv(dictionary, sep = '\n', header = None)
        n = list(d[0])
        self.train = pd.read_csv(train, sep = ' ', names = n)
        self.test = pd.read_csv(test, sep = ' ', names = n)
        self.w = np.zeros(shape= (self.train.shape[0]+1,self.train.shape[1]-1))



    
    def subset(self, df, first, second):
        label_name = list(df)[-1]
        sub = df[ (df[label_name] == first) | (df[label_name] == second)]

        return sub

    def training(self, first_val, second_val, first = False):
        train = self.subset(self.train, first_val, second_val).as_matrix()
        if first == True:
            self.w = np.zeros(shape= (train.shape[0]+1,train.shape[1]-1))
        else:
            self.w[0] = self.w[-1]


        for i in range(train.shape[0]):
            y = train[i][-1]
            if y != 1:
                y = -1
            x = train[i][:-1]
            if y * np.dot(x,self.w[i]) <= 0:
                self.w[i+1] = self.w[i] + y*x
            else:
                self.w[i+1] = self.w[i]
    def trainone(self, label):
        train = self.train.copy().as_matrix()
        w = np.zeros(shape= (train.shape[0]+1,train.shape[1]-1))
        

        for i in range(train.shape[0]):
            y = train[i][-1]
            if y != label:
                y = -1
            else:
                y = 1
            x = train[i][:-1]
            if y * np.dot(x,w[i]) <= 0:
                w[i+1] = w[i] + y*x
            else:
                w[i+1] = w[i]
        self.kclass[label] = w[-1]

    
    def error(self, data, first, second):
        test = self.subset(data,first,second).as_matrix()
        
        
        inc = 0.0 
        for i in range(test.shape[0]): 
            y = np.dot(self.w[-1],test[i][:-1])
            if(test[i][-1] > 1):
                label = -1
            else:
                label = test[i][-1]
            if (y < 0 and label > 0) or (y>0 and label <0):
                inc += 1.0

        return float(inc/test.shape[0])
    
