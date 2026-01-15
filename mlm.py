"""Minimal Learning Machine classes for regression and classification."""
import numpy as np
from scipy import fftpack
from scipy.spatial.distance import cdist
from scipy.optimize import least_squares

# from fcmeans import FCM
from mrsr import MRSR

from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.preprocessing import LabelBinarizer
from scipy.ndimage.filters import gaussian_filter1d
from scipy.stats import mode

class ERRORS():
    def not_train(self, obj):
        try:
                getattr(obj, "B")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")

def pinv_(X):
    try:
        return np.linalg.inv(X.T @ X) @ X.T
    except Exception as e:
        return np.linalg.pinv(X)


# MLM for regression (MLM): https://doi.org/10.1016/j.neucom.2014.11.073
class MLM(BaseEstimator, RegressorMixin):
    def __init__(self, rp_number=None, random_state=42):
        # random state
        self.random_state = random_state
        # number of reference points
        self.rp_number = rp_number
        #    if None, set rp_number to 10% of samples,
        #    if rp_number in [0,1], use as percentual.
        if self.rp_number == None: self.rp_number = 0.1

    def select_RPs(self):
        # random selection
        #    if <rp_number> equals to <N> use all points of RPs,
        #    else, select <rp_number> points at random.
        N = self.X.shape[0]

        if self.rp_number <= 1:    self.rp_number = int(self.rp_number * N)

        if self.rp_number == N:
            rp_id     = np.arange(N)
        else:
            r = np.random.RandomState(self.random_state)
            rp_id     = r.choice(N, self.rp_number, replace=False)

        self.rp_X     = self.X[rp_id,:]
        self.rp_y     = self.y[rp_id,:]

        self.D_x = cdist(self.X,self.rp_X)
        self.D_y = cdist(self.y,self.rp_y)

    def fit_B(self):
        self.B        = pinv_(self.D_x) @ self.D_y

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.select_RPs()
        self.fit_B()
        self.X_red = 1 - self.B.shape[0] / self.X.shape[0]
        self.y_red = 1 - self.B.shape[1] / self.y.shape[0]
        # delattr(self, 'X')
        # delattr(self, 'y')
        # delattr(self, 'D_x')
        return self

    def predict(self, X, y=None):
        return np.array([self.get_output(x) for x in X])

    def get_output(self, x):
        J = lambda y: self.in_cost(y, x)
        out = least_squares(J, x0=self.rp_y.mean(axis=0), method='lm')
        return out.x

    def in_cost(self, y, x):
        """internal cost function"""
        # make y a vector
        y  = np.array([y])

        # compute pairwise distance vectors
        #  - d_in: input space
        #  - d_out: output space
        d_x  = cdist(x[np.newaxis],self.rp_X)
        d_y  = cdist(y,self.rp_y)

        # compute the internal cost function
        # print(((d_y**2 - (d_x @ self.B)**2) / np.abs(d_y))[0])
        return ((d_y**2 - (d_x @ self.B)**2)**2)[0]

    def plot(self,plt,X=None, y=None, figsize=None):
        # X = X if X != None else self.X
        # y = y if y != None else self.y

        X_ = np.linspace(X.min(), X.max(), 300)[np.newaxis].T
        y_ = self.predict(X_)

        if X.shape[1] == 1:
            fig = plt.figure(figsize=figsize) if figsize != None else plt.figure()
            plt.scatter(X,y, marker='o', c='orange')
            plt.scatter(self.rp_X[:,0],self.rp_y[:,0],alpha=0.7,edgecolors='black',s=60,linewidths=2)
            plt.plot(X_, y_, c='black')
        else:
            print("X have more that one dimensions.")

