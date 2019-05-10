import numpy as np


class RndRegressor:

    def predict(self, x):

        return np.random.rand(len(x))

    def fit(self, x):
        return

    """def score(self, x, y):
        print(type(x))"""