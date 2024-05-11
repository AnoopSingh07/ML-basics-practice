import numpy as np

class LR():
    def __init__(self,N):
        self.theta= np.zeros(N)

    def train(self,X,Y):
        XT=np.transpose(X)
        bracket1= np.linalg.inv(np.dot(XT,X))
        bracket2=np.dot(XT,Y)
        self.theta=np.dot(bracket1,bracket2)
    
    def predict(self,X):
        return np.dot(self.theta,X)
    
obj=LR(2)
obj.train([[1,1],[2,3]],[4,8])
print(obj.predict([7,8]))
    
