from perceptron import Perceptron
p = Perceptron('pa3train.txt', 'pa3test.txt', 'pa3dictionary.txt')
p.opt_training(1,2)

print('second pass')
p.opt_training(1,2)

print('third pass')
p.opt_training(1,2)
print('\n')

p.print_w()
