import numpy as np
from utils.transform import standardize

with open('data/iris/iris.data', 'r') as f:
    data = f.readlines()

data = data[:-1]
data = [line.strip().split(',') for line in data]

features = np.array(list(map(lambda x : list(map(float, x)), [line[:-1] for line in data])))
targets = np.array([line[-1] for line in data])

features = standardize(features)

with open('data/iris/processed.data', 'w') as f:
    for i in range(len(features)):
        f.write(','.join(map(str, features[i])) + f',{targets[i]}' +'\n')
    f.write('\n')