import numpy as np
import hashlib

class IdealRegressor:

    def __init__(self, feature_vectors, targets, error_rate=0):
        self.hash_table = {}
        self.fit(feature_vectors, targets)
        self.error_rate = error_rate

    def fit(self, feature_vectors, targets):
        index = 0
        for vector in feature_vectors:
            hash_key = hashlib.md5(str(vector).encode('utf-8')).hexdigest()
            self.hash_table[hash_key] = targets[index]
            index += 1

    def predict(self, x_set):
        output = [self.find(x) for x in x_set]
        return output

    def find(self, x):
        hash_key = hashlib.md5(str(x).encode('utf-8')).hexdigest()
        if hash_key in self.hash_table:
            output = self.hash_table[hash_key]
            if output > 0.45:
                if np.random.rand() < self.error_rate:
                    return 1-output
            else:
                if np.random.rand() < self.error_rate:
                    return 1-output
            return output
        print('Unkown sample:', x)
        raise Exception('Unknown sample!')
