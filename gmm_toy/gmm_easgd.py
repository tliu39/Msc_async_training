import numpy as np
import matplotlib.pyplot as plt
from GaussianMixtureModel import GaussianMixureModel as GMM


# configure settings for figures globally
fontsize = 64
plt.rcParams["figure.figsize"] = (25, 20)
plt.rcParams["axes.labelsize"] = fontsize
plt.rcParams["xtick.labelsize"] = fontsize
plt.rcParams["ytick.labelsize"] = fontsize
plt.rcParams['lines.linewidth'] = 5
plt.rcParams["legend.fontsize"] = fontsize


# train
q = np.array([[0.6, 2, 4, 0.25, 5],
              [0.6, 2, 4, 0.25, 5]])
traj, t = GMM(same_target_dist=True).easgd_train(q, GMM.sgd, num_iterations=20*10**3, tau=10, learning_rate=0.25, moving_rate=0.01, batch_size=32)

folder = "easgd_std/"
# a1
for _ in range(traj.shape[0] - 1):
    #print(np.arange(t[_]), traj[_, :])
    plt.plot(traj[_, 0, :], alpha = 0.5, label="worker"+str(_))
central_traj = traj[-1, 0, :]
plt.plot(central_traj, label="master")
plt.legend()
plt.xlabel("iteration")
plt.ylabel("a1")
plt.savefig(folder + "a1.png")
plt.close()

# mu1
plt.plot(np.zeros(traj.shape[2]), label="ground truth")
for _ in range(traj.shape[0] - 1):
    #print(np.arange(t[_]), traj[_, :])
    plt.plot(traj[_, 1, :], alpha = 0.5, label="worker"+str(_))
central_traj = traj[-1, 1, :]
plt.plot(central_traj, label="master")
plt.legend()
plt.ylim(-1, 3.0)
plt.xlabel("iteration")
plt.ylabel("mu1")
plt.savefig(folder + "mu1.png")
plt.close()

# mu2
plt.plot(np.ones(traj.shape[2]) * 6, label="ground truth")
for _ in range(traj.shape[0] - 1):
    #print(np.arange(t[_]), traj[_, :])
    plt.plot(traj[_, 2, :], alpha = 0.5, label="worker"+str(_))
central_traj = traj[-1, 2, :]
plt.plot(central_traj, label="master")
plt.legend()
plt.xlabel("iteration")
plt.ylabel("mu2")
plt.savefig(folder + "mu2.png")
plt.close()

# sigma1
plt.plot(np.ones(traj.shape[2]), label="ground truth")
for _ in range(traj.shape[0] - 1):
    #print(np.arange(t[_]), traj[_, :])
    plt.plot(traj[_, 3, :], alpha = 0.5, label="worker"+str(_))
central_traj = traj[-1, 3, :]
plt.plot(central_traj, label="master")
plt.legend()
plt.ylim(-1, 3.0)
plt.xlabel("iteration")
plt.ylabel("sigma1")
plt.savefig(folder + "sigma1.png")
plt.close()

# sigma2
plt.plot(np.ones(traj.shape[2]) * 2, label="ground truth")
for _ in range(traj.shape[0] - 1):
    #print(np.arange(t[_]), traj[_, :])
    plt.plot(traj[_, 4, :], alpha = 0.5, label="worker"+str(_))
central_traj = traj[-1, 4, :]
plt.plot(central_traj, label="master")
plt.legend()
plt.xlabel("iteration")
plt.ylabel("sigma2")
plt.savefig(folder + "sigma2.png")
plt.close()