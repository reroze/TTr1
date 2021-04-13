import numpy as np

def CS(A):
    s = np.eandom.normal


A = np.array([[[1,2,3,4]
               ,[2,3,4,5]
               ,[3,4,5,6]]
              ,[[4,5,6,7],
                [5,6,7,8],
                [6,7,8,9]]])#2,3,4
#print(A.shape)
B_d = 100
B = np.random.normal(size=(B_d,B_d,B_d))
print('B.shape:', B.shape)
#A_shape=1
c = 125
B_shape=1
print(B.shape)
for i in range(B.ndim):
    B_shape *= B.shape[i]
print('B_shape', B_shape)
s = np.random.normal(size=[B_shape, 1])
#print(s)
#print(s.shape)
s = np.sign(s)
h = np.random.randint(low=0, high=c, size=B_shape)
y = np.zeros([c, 1])
#print(B.reshape(1,1,-1)[0])
#print(B.reshape(-1))
for i in range(B_shape):
    y[h[i]]+=s[i]*(B.reshape(-1))[i]

print(y.shape)

#B2 = np.zeros([100,100,100])
B2 = np.zeros([1000000, 1])
for i in range(1000000):
    B2[i]=s[i]*y[h[i]]
    #print(i)

print('B', B[0][0])
B3 = B-B2.reshape([100,100,100])
print('B3',B3[0][0])
B3 = B3**2
B3_num = B3.sum()
B3_num = B3_num**0.5

B = B**2
B_num = B.sum()
B_num = B_num**0.5
print(B3_num/B_num)
print('上确界', B_num/8000)
#print(s)
#print(s)


