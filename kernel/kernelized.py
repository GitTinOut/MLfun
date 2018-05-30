import pandas as pd
import numpy as np
class Perceptron:
    def __init__(self, train, test, p):
        self.train = pd.read_csv(train, sep = ' ', header = None).as_matrix()
        self.test = pd.read_csv(test, sep = ' ', header = None).as_matrix()
        self.p = p
        self.indices = np.zeros(shape = (1, self.train.shape[0]+1))
        
    def Kp(self, s, t):
        count = 0
        p = self.p
        for i in range(len(s) - p + 1):
            if s[i:i+p] in t:
                count += 1
            occured = False
            """
            for k in range(i):
                
                if v == s[k:k+p]:
                    occured = True
            if occured == True:
                continue
            """
        for j in range(len(t)-p+1):
            if t[j:j+p] in s:
                count += 1
        return count
    # < Wt, Phi(xt) >
    def WtdotPhi(self, t, x):
        train = self.train
        s = self.indices
        w = 0
        for i in range(t):
            w += s[0][i] * train[i][1] * self.Kp(self.train[i][0],x)
        return w

    def training(self):
        train = self.train

        for i in range(train.shape[0]):
            print(i)
            y = train[i][-1]
            if i == 0 or y * self.WtdotPhi(i-1,train[i][0]) <= 0:
                self.indices[0][i] += 1

    
    def error(self, data):
        test = data
        t = self.train.shape[0]
        inc = 0.0 
        for i in range(test.shape[0]): 
            if self.sign(self.WtdotPhi(t, test[i][0])) != test[i][1]:
                inc += 1

        return float(inc/test.shape[0])

    def prints(self):
        s = self.indices
        for i in s[0]:
            print(i)
    
    def set_p(self,p):
        self.p = p
    def sign(self, x):
        if x < 0:
            return -1
        else:
            return 1
