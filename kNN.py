
import numpy as np
from sklearn.base import BaseEstimator,ClassifierMixin
from scipy.spatial import distance
from scipy.stats import mode

class kNN(BaseEstimator, ClassifierMixin): 
  def __init__(self, n_neighbors:int = 5):
    self.n_neighbors = n_neighbors
    self.X_train = None
    self.y_train = None

  def fit(self, X, y):
    self.X_train = np.copy(X)
    self.y_train = np.copy(y)
    return self

  def predict(self, X):
    distances = distance.cdist(XA=X, XB=self.X_train, metric='euclidean')
    nn = np.argpartition(a=distances, kth=self.n_neighbors, axis=1)[:, :self.n_neighbors]
    return mode(self.y_train[nn], axis=1)[0].flatten()
