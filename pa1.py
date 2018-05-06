from knn import KNN
import pandas as pd
import numpy as np
import datetime

model = KNN('pa1train.txt', 'pa1validate.txt', 'pa1test.txt', k = 1)

print('k = 1')
ti = datetime.datetime.now()
model.predicttrain()
model.predictvalid()
model.predicttest()
tf = datetime.datetime.now()
print(tf-ti)



print('k = 5')
ti = datetime.datetime.now()
model.setk(5)
model.predicttrain()
model.predictvalid()
model.predicttest()
tf = datetime.datetime.now()
print(tf-ti)

print('k = 9')
ti = datetime.datetime.now()
model.setk(9)
model.predicttrain()
model.predictvalid()
model.predicttest()
tf = datetime.datetime.now()
print(tf-ti)

print('k = 15')
ti = datetime.datetime.now()
model.setk(15)
model.predicttrain()
model.predictvalid()
model.predicttest()
tf = datetime.datetime.now()
print(tf-ti)


#projection optimization
print("Projection Results")
model.setp('projection.txt')

print('k = 1')
ti = datetime.datetime.now()
model.setk(1)
model.predicttrain(proj = True)
model.predictvalid(proj = True)
model.predicttest(proj = True)
tf = datetime.datetime.now()
print(tf-ti)


print('k = 5')
ti = datetime.datetime.now()
model.setk(5)
model.predicttrain(proj = True)
model.predictvalid(proj = True)
model.predicttest(proj = True)
tf = datetime.datetime.now()
print(tf-ti)

print('k = 9')
ti = datetime.datetime.now()
model.setk(9)
model.predicttrain(proj = True)
model.predictvalid(proj = True)
model.predicttest(proj = True)
tf = datetime.datetime.now()
print(tf-ti)

print('k = 15')
ti = datetime.datetime.now()
model.setk(15)
model.predicttrain(proj = True)
model.predictvalid(proj = True)
model.predicttest(proj = True)
tf = datetime.datetime.now()
print(tf-ti)
