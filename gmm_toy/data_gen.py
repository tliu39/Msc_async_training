import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# configure settings for figures globally
fontsize = 64
plt.rcParams["figure.figsize"] = (20, 20)
plt.rcParams["axes.labelsize"] = 40
plt.rcParams["xtick.labelsize"] = fontsize
plt.rcParams["ytick.labelsize"] = fontsize
plt.rcParams['lines.linewidth'] = 5
plt.rcParams["legend.fontsize"] = fontsize

# Initialize 2 data clusters with different number of samples and two distinct labels
d1 = np.zeros((4000, 2))
d2 = np.ones((6000, 2))

# Sample the training data from 2 normal distributions
d1[:, 0] = np.random.normal(loc=0, scale=1, size=d1.shape[0])
d2[:, 0] = np.random.normal(loc=6, scale=2, size=d2.shape[0])

# Save the data
np.save("data/data1", d1)
np.save("data/data2", d2)

# Plot the distribution of the training data for visualization and save the figure
sns.distplot(np.load("data/data1.npy")[:, 0], label="Data Cluster 1")
sns.distplot(np.load("data/data2.npy")[:, 0], label="Data Cluster 2")
plt.legend()
plt.savefig("data/visualization.png")
plt.close()
