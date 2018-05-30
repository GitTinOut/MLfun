from kernelized import Perceptron
p = Perceptron('pa4train.txt', 'pa4test.txt', p = 2)
p.training()
print(p.error(p.train))
#print(p.prints())

s = 'asdf'
t= 'gpsd'
p = Perceptron('pa4train.txt', 'pa4test.txt', p = 2)
p.set_p(1)
