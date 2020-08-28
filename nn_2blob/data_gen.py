import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns


# configure settings for figures globally
fontsize = 64
plt.rcParams["figure.figsize"] = (20, 20)
plt.rcParams["lines.markersize"] = 15
plt.rcParams["axes.labelsize"] = fontsize
plt.rcParams["xtick.labelsize"] = fontsize
plt.rcParams["ytick.labelsize"] = fontsize
plt.rcParams['lines.linewidth'] = 5
plt.rcParams["legend.fontsize"] = fontsize
plt.rcParams["grid.alpha"] = 0.75

num_sample = 64

mu0 = np.array([-3, -8])
sigma0 = np.array([[1, 0], [0, 16]])
cluster0 = np.random.default_rng().multivariate_normal(mu0, sigma0, size=num_sample)
plt.scatter(cluster0[:, 0], cluster0[:, 1], label="cluster 0")

mu1 = np.array([3, 8])
sigma1 = np.array([[1, 0], [0, 16]])
cluster1 = np.random.default_rng().multivariate_normal(mu1, sigma1, size=num_sample)
plt.scatter(cluster1[:, 0], cluster1[:, 1], label="cluster 1")

plt.legend(loc="upper left")
plt.savefig("visualization/data.png")
plt.close()

# plt.scatter(cluster0[:, 0], cluster0[:, 1], label="cluster 0")
# plt.scatter(cluster1[:, 0], cluster1[:, 1], label="cluster 1")
# plt.legend()
# plt.show()
# plt.close()

# Putting data together
data0 = np.hstack((cluster0, np.zeros((num_sample, 1))))
data1 = np.hstack((cluster1, np.ones((num_sample, 1))))
data = np.vstack((data0, data1))

# normalize data
# scaler = StandardScaler()
# data[:, 0:2] = scaler.fit_transform(data[:, 0:2])
#
# plt.scatter(data[:num_sample, 0], data[:num_sample, 1], label="cluster 0")
# plt.scatter(data[num_sample:, 0], data[num_sample:, 1], label="cluster 1")
# plt.legend()
# plt.savefig("visualization/normal_data.png")
# plt.close()
#
# plt.scatter(data[:num_sample, 0], data[:num_sample, 1], label="cluster 0")
# plt.scatter(data[num_sample:, 0], data[num_sample:, 1], label="cluster 1")
# plt.legend()
# plt.show()
# plt.close()

# Shuffle data
rng = np.random.default_rng()
rng.shuffle(data)
np.save("data", data)

test_sample = 16
test_cluster0 = np.hstack((np.random.default_rng().multivariate_normal(mu0, sigma0, size=test_sample), np.zeros((test_sample, 1))))
test_cluster1 = np.hstack((np.random.default_rng().multivariate_normal(mu1, sigma1, size=test_sample), np.ones((test_sample, 1))))
test = np.vstack((test_cluster0, test_cluster1))
np.save("test", test)
