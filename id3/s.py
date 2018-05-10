from DTree import DTree
import pandas as pd
import numpy as np
"""
d = DTree(None, None, None)
x = [0,1,0,2,0,2,0]
y= [1,1,1,0,0,0,0]
print(d.s_entropy_of_feat(x,y))

d = DTree('pa2train.txt', 'pa2validation.txt', 'pa2test.txt')
df = pd.read_csv('pa2train.txt', header =None, sep =' ')
m = df.as_matrix()
l, r = d.split(m, 22, .5)
x = 0
for i in range(l.shape[0]):
    if l[i][22] != 0:
        x = 1
print(x==0)
print(l)
print(r)"""
d = DTree('pa2train.txt', 'pa2validation.txt', 'pa2test.txt')
d.buildTree()
print(d.test_train())
