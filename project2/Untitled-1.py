import numpy as np
import matplotlib.pyplot as plt

import statsmodels.api as sm

class LinearRegression:
    """
    X: an exogenous variable is one whose value is determined outside the model and is imposed on the model
    y: an endogenous variable is a variable whose value is determined by the model
    """
    def __init__(self, X, y): # 생성자
        self.X = X # exogenous variable
        self.y = y # endogenous variable
        self.rank = None # rank of the design matrix X
        self._dof_model = None # model degrees of freedom 
        self._dof_resid = None # residual degrees of freedom
        self.nob = self.X.shape[0] # number of observations
        self.dim = self.X.shape[1] # dimension of the data (in Hayashi, dim refers to K)
        self.Q = None
        self.R = None

    def dof_model(self):
        """
        model degrees of freedom is defined by:

        rank of X - 1
        """
        self._dof_model = self.rank_X() - 1
        return self._dof_model

    def set_dof_model(self, value):
        self._dof_model = value
    
    def dof_resid(self):
        """
        residual degrees of freedom is defined by:

        # observations - rank of X
        """
        self._dof_resid = self.nob - self.rank_X()
        return self._dof_resid 

    def rank_X(self):
        if self.rank == None:
            self.rank = np.linalg.matrix_rank(self.X)
        return self.rank

    def set_dof_resid(self, value):
        self._dof_resid = value

    def fit(self, method = "qr"):
        if method == "qr":
            Q, R = np.linalg.qr(self.X)
            effect = np.dot(Q.T, y)
            beta = np.linalg.solve(R, effect)
            self.Q, self.R = Q, R
        return beta

    def predict(self, _X):
        y_pred = np.dot(_X, self.fit())
        return y_pred 

nsample = 100
x = np.linspace(0, 10, 100)
X = np.column_stack((x, x**2))
beta = np.array([1, 0.1, 10])
e = np.random.normal(size=nsample)

X = sm.add_constant(X)
y = np.dot(X, beta) + e

_X = np.array([1, 1, 1])

lm = LinearRegression(X, y)
lm.fit()

print(lm.fit())
print(lm.predict(_X))

print( 0.80946252+  0.05857567+ 10.00567566)

