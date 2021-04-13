import numpy as np
a = [1, 1, -1]
a = np.array(a)
b = [[1, 1, 2],
     [2, 2, 4]]
b = np.array(b)
print(a @ b.T)
