from perceptron import Perceptron

p = Perceptron('pa3train.txt', 'pa3test.txt', 'pa3dictionary.txt')
p.training(1, 2)
p.opt_training(1,2)
print(p.error(p.train,1,2))
print(p.v_error(p.train,1,2))
print(p.a_error(p.train,1,2))
