import pandas as pd
import numpy as np
class Perceptron:
    def __init__(self, train, test, dictionary):
        d = pd.read_csv(dictionary, sep = '\n', header = None)
        n = list(d[0])
        n.append('class')
        self.train = pd.read_csv(train, sep = ' ', names = n)
        self.test = pd.read_csv(test, sep = ' ', names = n)
        self.w = np.zeros(shape= (self.train.shape[0]+1,self.train.shape[1]-1))
        self.ow = np.zeros(shape= (self.train.shape[0],self.train.shape[1]-1))
        self.m = 1
        self.c = np.ones(shape = (self.train.shape[0],1))
        self.c[0] = 1

        self.kclass = np.zeros(shape = (self.train.shape[0]+1, self.train.shape[1]-1))

        self.dictionary = d[0]

        self.confusion = np.zeros(shape = (7,6))
        
    
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

    def opt_training(self, first_val, second_val):
        train = self.subset(self.train, first_val, second_val).as_matrix()
        for t in range(train.shape[0]):
            y = train[t][-1]
            if y != 1:
                y = -1

            x = train[t][:-1]
            if y * np.dot(x,self.ow[self.m]) <= 0:
                self.ow[self.m+1] = self.ow[self.m] + y*x
                self.m = self.m+1
                self.c[self.m] = 1
            else:
                self.c[self.m] += 1
    def confusion_matrix(self,):
        test = self.test.copy().as_matrix()
        Ci = self.kclass
        for k in range(test.shape[0]):
            actual_label = test[k][-1]
            predicted_label = -1
            total = 0
            for i in range(1,7):
                y = np.dot(Ci[i],test[k][:-1])
                if y > 0:
                    y = 1
                    predicted_label = i
                    total += 1
                else:
                    y = -1
            if total == 1:
                self.confusion[predicted_label-1][actual_label-1] += 1
            else:
                self.confusion[6][actual_label-1] +=1
        for j in range(1,7):
            df = self.test.copy()
            Nj = len(df[df['class'] == j])
            for i in range(1,8):
                self.confusion[i-1][j-1] /= Nj
        print(self.confusion)



    
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
    
    def v_error( self, data, first, second):
        test = self.subset(data, first, second).as_matrix()

        inc = 0.0
        for i in range(test.shape[0]):
            y = 0
            label = test[i][-1]
            if label > 1:
                label = -1
            for j in range(self.m):
                r = np.dot(self.ow[j],test[i][:-1])
                if r >= 0:
                    r = 1
                elif r < 0:
                    r = -1
                #print(r)
                y += self.c[j]*r
            #print(y)
            #print(label)
            if (y < 0 and label > 0) or (y>0 and label <0):
                inc += 1.0
            

        return float(inc/test.shape[0])

    def a_error( self, data, first, second):
        test = self.subset(data, first, second).as_matrix()

        inc = 0.0
        for i in range(test.shape[0]):
            y = 0
            label = test[i][-1]
            if label > 1:
                label = -1 
            for j in range(self.m):
                y += self.c[j]*self.ow[j]
            y = np.dot(y, test[i][:-1])
            #print(y)
            #print(label)
            if (y < 0 and label > 0) or (y>0 and label <0):
                inc += 1.0
            

        return float(inc/test.shape[0])
    def print_w(self):
        #print(self.ow.shape)
        df = pd.DataFrame(self.ow, columns = self.dictionary)
        #print(df.shape)
        #print(df)
        i = self.dictionary[2]
        #print( len(df[df[i]<0][i]))
        val2words = {}
        for i in self.dictionary:
            count = len(df[df[i] > 0][i])
            if count == 0:
                continue
            val2words[count] = i
        neg2pos = sorted(val2words.keys())
        print('most positive words')
        print(neg2pos[-1])
        print(val2words[neg2pos[-1]])
        print(neg2pos[-2])
        print(val2words[neg2pos[-2]])
        print(neg2pos[-3])
        print(val2words[neg2pos[-3]])

        val2words = {}
        for i in self.dictionary:
            count = len(df[df[i] < 0][i])
            if count == 0:
                continue
            val2words[count] = i
        neg2pos = sorted(val2words.keys())
        print('most negative words')
        print(neg2pos[-1])
        print(val2words[neg2pos[-1]])
        print(neg2pos[-2])
        print(val2words[neg2pos[-2]])
        print(neg2pos[-3])
        print(val2words[neg2pos[-3]])


