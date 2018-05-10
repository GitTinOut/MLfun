class Node:
    def __init__(self, data, f = None, t = None, pure = False, label = None ):
        self.data = data
        self.left = None
        self.right = None
        self.f = f
        self.t = t
        self.pure = pure
        self.label = label
    
    def setLeft(self, left):
        self.left = left
        
    def setRight(self, right):
        self.right = right
    def setT(self, t):
        self.t = t
    def setF(self, f):
        self.f = f

    def getData(self):
        return self.data

    def getNext(self, v):
        if v[f] < t:
            return self.left
        else:
            return self.right
    def getT(self):
        return self.t

    def getF(self):
        return self.f
    
    def isPure(self):
        return self.pure
    
    def getLabel(self):
        return self.label
