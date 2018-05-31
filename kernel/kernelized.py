import pandas as pd
from collections import Counter
from collections import defaultdict
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
        """o_s = dict()
        o_t = dict()
        for i in range(len(s) - p + 1):
            sub = s[i:i+p]
            if sub not in o_s:
                o_s[sub] = 1
            else:
                o_s[sub] += 1
        
        for j in range(len(t)-p+1):
            sub = t[j:j+p]
            if sub not in o_t:
                o_t[sub] = 1
            else:
                o_t[sub] += 1
        intersection = list(set(o_s).intersection(set(o_t)))

        for i in intersection:
            count += o_s[i] * o_t[i]
        """
        s_list = []
        for i in range(len(s)-p+1):
            s_list.append(s[i:i+p])

        s_set = Counter(s_list)
        
        t_list = []
        for i in range(len(t)-p+1):
            t_list.append(t[i:i+p])
        t_set = Counter(t_list)

        for word in t_set:
            if (word in s_set):
                count += s_set[word] * t_set[word]
        
        return count

    def Mp(self, s, t):
        count = 0
        p = self.p
        s_list = []
        for i in range(len(s)-p+1):
            s_list.append(s[i:i+p])

        s_set = Counter(s_list)
        
        t_list = []
        for i in range(len(t)-p+1):
            t_list.append(t[i:i+p])
        t_set = Counter(t_list)

        for word in t_set:
            if (word in s_set):
                count += 1 
        
        return count
    # < Wt, Phi(xt) >
    def WtdotPhi(self, t, x):
        train = self.train
        s = self.indices
        w = 0
        for i in range(t):
            if s[0][i] == 0:
                continue
            xi = train[i][0]
            y = train[i][1]
            w += y * self.Kp(xi,x)
        return w
    # < Wt, Phi(xt) >
    def WtdotPhi2(self, t, x):
        train = self.train
        s = self.indices
        w = 0
        for i in range(t):
            if s[0][i] == 0:
                continue
            xi = train[i][0]
            y = train[i][1]
            w += y * self.Mp(xi,x)
        return w

    def training(self):
        train = self.train

        for i in range(train.shape[0]):
            x = train[i][0]
            y = train[i][-1]
            if i == 0 or y * self.WtdotPhi(i,x) <= 0:
                self.indices[0][i] += 1
    def training2(self):
        train = self.train

        for i in range(train.shape[0]):
            x = train[i][0]
            y = train[i][-1]
            if i == 0 or y * self.WtdotPhi2(i,x) <= 0:
                self.indices[0][i] += 1
        

    
    def error(self, data):
        test = data
        t = self.train.shape[0]
        inc = 0.0 
        for i in range(test.shape[0]): 
            x = test[i][0]
            y = test[i][1]
            if y * self.WtdotPhi(t, x) <= 0:
                inc += 1

        return float(inc/test.shape[0])

    def prints(self):
        s = self.indices
        for i in s[0]:
            print(i)
    
    def set_p(self,p):
        self.p = p
    def substrings(self):
        s = self.indices
        data = self.train
        p = 5
        substr = defaultdict()
        for i in range(data.shape[0]):
            if s[0][i] == 0:
                continue

            x = data[i][0]
            y = data[i][1]
            s_list = []
            for j in range(len(x)-p+1):
                s_list.append(x[j:j+p])
            s_map = Counter(s_list)
            for i in s_map:
                if i not in substr:
                    substr[i] = y*s_map[i]
                else:
                    substr[i] += y*s_map[i]
        print(substr)
        print(max(substr, key=substr.get))
        del substr[max(substr, key=substr.get)]
        print(max(substr, key=substr.get))
        del substr[max(substr, key=substr.get)]
        print(max(substr, key=substr.get))
        del substr[max(substr, key=substr.get)]
        print(max(substr, key=substr.get))
        del substr[max(substr, key=substr.get)]
        print(max(substr, key=substr.get))
        del substr[max(substr, key=substr.get)]
        """ 
        s_list = []
        for i in range(len(s)-p+1):
            s_list.append(s[i:i+p])

        s_set = Counter(s_list)
        """

