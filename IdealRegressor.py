import numpy as np
import hashlib

class IdealRegressor:

    def __init__(self, feature_vectors, targets):
        self.feature_vectors = feature_vectors
        self.targets = targets
        index = 0
        self.hash_table = {}
        for vector in self.feature_vectors:
            hash_key = hashlib.md5(str(vector).encode('utf-8')).hexdigest()
            self.hash_table[hash_key] = self.targets[index]
            index += 1


    def predict(self, x_set):
        output = [self.find(x) for x in x_set]
        return output

    def find(self, x):
        hash_key = hashlib.md5(str(x).encode('utf-8')).hexdigest()
        if hash_key in self.hash_table:
            return self.hash_table[hash_key]
        raise Exception('Unknown sample!')
