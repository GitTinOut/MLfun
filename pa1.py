from knn import KNN

model = KNN('pa1train.txt', 'pa1validate.txt', 'pa1test.txt', k = 1)
print('k = 1')
model.predicttrain()
model.predictvalid()

print('k = 5')
model.setk(5)
model.predicttrain()
model.predictvalid()

print('k = 9')
model.setk(9)
model.predicttrain()
model.predictvalid()

print('k = 15')
model.setk(15)
model.predicttrain()
model.predictvalid()
