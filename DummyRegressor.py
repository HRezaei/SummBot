import numpy as np


class RndRegressor:

    def predict(self, x):

        return np.random.rand(len(x))
