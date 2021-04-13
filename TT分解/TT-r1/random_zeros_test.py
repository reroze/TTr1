import numpy as np

I = 60
J = 60
K = 60

tensor = np.random.normal(size=[I, J, K])
tensor2 = np.zeros([I, J, K])
print(tensor.shape)
print(tensor2.shape)
