import numpy as np
import matplotlib.pyplot as plt

# configure settings for figures globally
fontsize = 64
plt.rcParams["figure.figsize"] = (25, 25)
plt.rcParams["lines.markersize"] = 7.5
plt.rcParams["axes.labelsize"] = fontsize
plt.rcParams["xtick.labelsize"] = fontsize
plt.rcParams["ytick.labelsize"] = fontsize
plt.rcParams['lines.linewidth'] = 5
plt.rcParams["legend.fontsize"] = fontsize
#plt.rcParams["grid.alpha"] = 0.75

N = 4500 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes
X = np.zeros((N*K,D)) # data matrix (each row = single example)
y = np.zeros(N*K, dtype='uint8') # class labels
a = 1
p = 0.5
c = 0.015
b = 2
for j in range(K):
  ix = range(N*j,N*(j+1))
  t = np.random.uniform(size=N) # t
  r = a * t ** p # radius
  # r = np.linspace(0, 1, N)
  #t = np.linspace(j*4,(j+4)*4,N) + np.random.randn(N)*0.2 # theta
  theta = 2 * b * t**p * np.pi + j * 2 * np.pi / K # theta
  X[ix] = np.c_[r*np.sin(theta) + np.random.randn(N) * c, r*np.cos(theta) + np.random.randn(N) * c]
  y[ix] = j
  plt.scatter(X[ix, 0], X[ix, 1], label="label "+str(j))
# lets visualize the data:
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
# plt.show()
# plt.close()

# plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.11), ncol=3)
plt.savefig("visualization/data.png")

np.save("data/sample", X)
np.save("data/label", y)

N = 500 # number of points per class
for j in range(K):
  ix = range(N*j,N*(j+1))
  t = np.random.uniform(size=N) # t
  r = a * t ** p # radius
  # r = np.linspace(0, 1, N)
  #t = np.linspace(j*4,(j+4)*4,N) + np.random.randn(N)*0.2 # theta
  theta = 2 * b * t**p * np.pi + j * 2 * np.pi / K # theta
  X[ix] = np.c_[r*np.sin(theta) + np.random.randn(N) * c, r*np.cos(theta) + np.random.randn(N) * c]
  y[ix] = j
np.save("data/Xtest", X)
np.save("data/Ytest", y)

