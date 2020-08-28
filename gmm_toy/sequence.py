import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# configure settings for figures globally
plt.rcParams["figure.figsize"] = (25, 10)
plt.rcParams["axes.labelsize"] = 40
plt.rcParams["xtick.labelsize"] = 40
plt.rcParams["ytick.labelsize"] = 40
plt.rcParams['lines.linewidth'] = 5

# load the data and also put them together
d1 = np.load("data/data1.npy")
d2 = np.load("data/data2.npy")
data = np.vstack((d1, d2))

# Extract the 2 types of training and label
# The first 2 sets of training and label contain data with same label
# d1_X = d1[:, 0]
# d2_X = d1[:, 0]
# d1_y = d1[:, 1]
# d2_y = d1[:, 1]

# The second 2 sets of training and label contain data with both labels
# X1 = np.vstack((d1[0:d1.shape[0]//2, 0], d2[0:d2.shape[0]//2, 0]))
# X2 = np.vstack((d1[d1.shape[0]//2:, 0], d2[d2.shape[0]//2:, 0]))
# y1 = np.vstack((d1[0:d1.shape[0]//2, 1], d2[0:d2.shape[0]//2, 1]))
# y2 = np.vstack((d1[d1.shape[0]//2:, 1], d2[d2.shape[0]//2:, 1]))


def U(x):
    """The potential function"""
    a1 = 0.4
    sigma1 = 1
    sigma2 = 2
    mu1 = 0
    mu2 = 6
    return a1/(sigma1 * np.sqrt(2 * np.pi)) * np.exp(-1/2 * ((x - mu1)/ sigma1)**2) + \
           (1-a1)/(sigma2 * np.sqrt(2 * np.pi)) * np.exp(-1/2 * ((x - mu2)/ sigma2)**2)

def Metropolis(q0, h):
    """Metropolis hasting"""
    R = np.random.normal()
    S = np.random.uniform()
    q = q0 + h * R
    if S < U(q) / U(q0):
        flag = 1
    else:
        q = q0
        flag = 0
    return q, flag

def run_simulation(q, replicas=2, Nsteps=10**4, tau=10, h=0.01, moving_rate=0.01, step_function=Metropolis):
    """define asynchronous processes"""
    traj = np.empty((replicas+1, Nsteps))
    traj[:, :] = np.nan
    traj[:replicas, 0] = np.copy(q)
    traj[replicas, 0] = np.mean(q)
    t = np.zeros(replicas, dtype=int)

    for i in range(1, Nsteps):
        if i % 10**3 == 0:
            print(i // 10 ** 3, "thousand iterations")
        pid = np.random.randint(replicas)
        x_i = traj[pid, t[pid]]
        x = x_i
        if t[pid] % tau == 0:
            x_bar = traj[-1, i-1]
            x_i -= moving_rate * (x - x_bar)
            x_bar += moving_rate * (x - x_bar)
            traj[-1, i] = x_bar
        else:
            traj[-1, i] = traj[-1, i-1]
        x_i, accepted = step_function(x_i, h)
        t[pid] += 1
        traj[pid, t[pid]] = x_i
    return traj, t

# run simulations
traj, t = run_simulation(np.array([-2.0, 10.0]), replicas=2, Nsteps=10**4, tau=2, h=0.1, moving_rate=0.01, step_function=Metropolis)
print(t)
for _ in range(traj.shape[0] - 1):
    print(np.arange(t[_]), traj[_, :t[_]])
    plt.plot(traj[_, :t[_]], alpha = 0.5, label="local_var"+str(_))
central_traj = traj[-1, np.arange(0, traj.shape[1], traj.shape[0] - 1)]
plt.plot(central_traj, label="central_var")
plt.legend()
plt.savefig("figures/farapart.png")
plt.close()
