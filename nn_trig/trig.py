import numpy as np
import matplotlib.pyplot as plt

# configure settings for figures globally
fontsize = 64
plt.rcParams["figure.figsize"] = (25, 25)
plt.rcParams["lines.markersize"] = 8
plt.rcParams["axes.labelsize"] = fontsize
plt.rcParams["xtick.labelsize"] = fontsize
plt.rcParams["ytick.labelsize"] = fontsize
plt.rcParams['lines.linewidth'] = 5
plt.rcParams["legend.fontsize"] = fontsize
plt.rcParams["grid.alpha"] = 0.5

N = 4500 # number of points per class
D = 2 # dimensionality
K = 2 # number of classes
X = np.zeros((N*K,D)) # data matrix (each row = single example)
y = np.zeros(N*K, dtype='uint8') # class labels
a = 1
p = 0.5
c = 0.015
b = 4

# class 0
ix = range(N*1)
t = np.random.uniform(size=N) # t
x1 = a * t
x2 = np.cos(b * t * np.pi) + c * np.random.randn(N)
X[ix] = np.c_[x1, x2]
y[ix] = 0
plt.scatter(X[ix, 0], X[ix, 1], label="label 0")

# class 1
ix = range(N*1, N*2)
t = np.random.uniform(size=N) # t
x1 = a * t
x2 = np.sin(b * t * np.pi) + c * np.random.randn(N)
X[ix] = np.c_[x1, x2]
y[ix] = 1
plt.scatter(X[ix, 0], X[ix, 1], label="label 1")

# # lets visualize the data:
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
# plt.show()
# plt.close()

#plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.legend(bbox_to_anchor=(0.1, 1.0), ncol=2)
plt.savefig("data/data.png")

np.save("data/sample", X)
np.save("data/label", y)

N = 400 # number of points per class
# class 0
ix = range(N*1)
t = np.random.uniform(size=N) # t
x1 = a * t
x2 = np.cos(b * t * np.pi) + c * np.random.randn(N)
X[ix] = np.c_[x1, x2]
y[ix] = 0

# class 1
ix = range(N*1, N*2)
t = np.random.uniform(size=N) # t
x1 = a * t
x2 = np.sin(b * t * np.pi) + c * np.random.randn(N)
X[ix] = np.c_[x1, x2]
y[ix] = 1
np.save("data/Xtest", X)
np.save("data/Ytest", y)

