"""import pandas as pd
from collections import Counter
from collections import defaultdict
import numpy as np
class Boost:
    def __init__(self, train, test, d):
        self.dict = list(pd.read_csv(d, sep = '\n', header = None)[0])
        self.train = pd.read_csv(train, sep = ' ', header = None).as_matrix()
        self.test = pd.read_csv(test, sep = ' ', header = None).as_matrix()

"""
from sklearn.ensemble import AdaBoostClassifier 
import pandas as pd

names = list(pd.read_csv('pa5dictionary.txt', sep = '\n', header = None)[0])
train = pd.read_csv('pa5train.txt', sep = ' ', header = None).as_matrix()
test = pd.read_csv('pa5test.txt', sep = ' ', header = None).as_matrix()

def train_and_error(t, train, test):

	clf = AdaBoostClassifier(n_estimators = t, learning_rate= .5, algorithm = "SAMME")
	print("t = " + str(t))
	clf.fit(train[:,:-1], train[:,-1])

	print("Training Error :"+ str(1 - clf.score(train[:,:-1], train[:,-1])))
	print( "Test Error: "+ str(1 - clf.score(test[:,:-1], test[:,-1])))

for i in [3,7,10,15,20]:
	train_and_error(i, train, test)

