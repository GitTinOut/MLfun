from kernelized import Perceptron
import datetime
"""
p = Perceptron('pa4train.txt', 'pa4test.txt', p = 2)
ti = datetime.datetime.now()
p.training()
print(p.error(p.train))
tf = datetime.datetime.now()
print(tf-ti)
#print(p.prints())
"""
p = Perceptron('pa4train.txt', 'pa4test.txt', p = 5)
p.training()
ti = datetime.datetime.now()
p.substrings()
tf = datetime.datetime.now()
print(tf-ti)
