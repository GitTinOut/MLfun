from kernelized import Perceptron
import datetime
p = Perceptron('pa4train.txt', 'pa4test.txt', p = 2)
ti = datetime.datetime.now()
p.training()
print(p.error(p.train))
tf = datetime.datetime.now()
print(tf-ti)
#print(p.prints())

