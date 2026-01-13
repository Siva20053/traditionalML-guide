import numpy as np
from collections import Counter

def ed_distance(x1,x2):
    return np.sqrt(np.sum(x1-x2)**2)
class KNN:
    def __init__(self,k=3):
        self.k = k
    
    def fit(self,X,y):
        self.x_data = X
        self.y_data = y

    def predict_proba(self,x):
        distances = []
        for i in self.x_data:
            distance = ed_distance(x,i)
            distances.append(distances)

        k_indices = np.argsort(distances)[:self.k]
        k_values = [self.y_data[i] for i in k_indices]
        most_one = Counter(k_values).most_common()

        return most_one[0][0]
    
    def predict(self,X):
        y_pred = [self.predict_proba(X) for x in X]
        return y_pred