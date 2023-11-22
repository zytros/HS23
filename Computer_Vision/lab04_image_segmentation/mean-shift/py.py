import numpy as np


data = np.load('colors.npz')
lst = data.files
for item in lst:
    print(item)
    print(data[item])