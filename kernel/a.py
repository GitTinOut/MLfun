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
p = Perceptron('pa4train.txt', 'pa4test.txt', p = 3)
ti = datetime.datetime.now()
p.training()
print('p=3')
print('train = ' + str(p.error(p.train)))
print('test = ' + str(p.error(p.test)))
tf = datetime.datetime.now()
print(tf-ti)

p = Perceptron('pa4train.txt', 'pa4test.txt', p = 4)
ti = datetime.datetime.now()
p.training()
print('p=4')
print('train = ' + str(p.error(p.train)))
print('test = ' + str(p.error(p.test)))
tf = datetime.datetime.now()
print(tf-ti)

p = Perceptron('pa4train.txt', 'pa4test.txt', p = 5)
ti = datetime.datetime.now()
p.training()
print('p=5')
print('train = ' + str(p.error(p.train)))
print('test = ' + str(p.error(p.test)))
tf = datetime.datetime.now()
print(tf-ti)
