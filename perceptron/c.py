from perceptron import Perceptron
p = Perceptron('pa3train.txt', 'pa3test.txt', 'pa3dictionary.txt')

for i in range(1,7):
    p.trainone(i)

p.confusion_matrix()
