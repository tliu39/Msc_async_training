import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# configure settings for figures globally
fontsize = 64
plt.rcParams["figure.figsize"] = (25, 12)
plt.rcParams["axes.labelsize"] = fontsize
plt.rcParams["xtick.labelsize"] = fontsize
plt.rcParams["ytick.labelsize"] = fontsize
plt.rcParams['lines.linewidth'] = 5
plt.rcParams["legend.fontsize"] = fontsize

# load the data and also put them together
d1 = np.load("data/data1.npy")
d2 = np.load("data/data2.npy")
data = np.vstack((d1, d2))

# Extract the 2 types of training and label
# The first 2 sets of training and label contain data with same label
d1_X = d1[:, 0]
d2_X = d2[:, 0]
d1_y = d1[:, 1]
d2_y = d2[:, 1]

# The second 2 sets of training and label contain data with both labels
X1 = np.hstack((d1[0:d1.shape[0]//2, 0], d2[0:d2.shape[0]//2, 0])).reshape(-1, 1)
X2 = np.hstack((d1[d1.shape[0]//2:, 0], d2[d2.shape[0]//2:, 0])).reshape(-1, 1)
y1 = np.hstack((d1[0:d1.shape[0]//2, 1], d2[0:d2.shape[0]//2, 1])).reshape(-1, 1)
y2 = np.hstack((d1[d1.shape[0]//2:, 1], d2[d2.shape[0]//2:, 1])).reshape(-1, 1)

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

def E(x, params):
    """Perform the Expectation step"""
    a1, mu1, mu2, sigma1, sigma2 = params
    numerator = a1/(sigma1 * np.sqrt(2 * np.pi)) * np.exp(-1/2 * ((x - mu1)/ sigma1)**2)
    denominator = numerator + (1-a1)/(sigma2 * np.sqrt(2 * np.pi)) * np.exp(-1/2 * ((x - mu2)/ sigma2)**2)
    l1 = numerator / denominator
    l2 = 1 - numerator / denominator
    return l1, l2

def M(x, l1, l2):
    """Perform the Maximization step"""
    mu1 = np.sum(l1 * x) / np.sum(l1)
    mu2 = np.sum(l2 * x) / np.sum(l2)
    sigma1 = np.sqrt( np.sum(l1 * (x - mu1) ** 2) / np.sum(l1) )
    sigma2 = np.sqrt( np.sum(l2 * (x - mu2) ** 2) / np.sum(l2) )
    a1 = np.sum(l1) / x.shape[0]
    a2 = np.sum(l2) / x.shape[0]
    return np.array([a1, mu1, mu2, sigma1, sigma2])

def run_simulation(q, X, replicas=2, Nsteps=10**4, tau=10, h=0.01, moving_rate=0.01):
    """define asynchronous processes"""
    traj = np.empty((replicas+1, 5, Nsteps))
    traj[:, :, :] = np.nan
    traj[:replicas, :, 0] = np.copy(q)
    traj[replicas, :, 0] = np.mean(q, axis=0)
    t = np.zeros(replicas, dtype=int)

    for i in range(1, Nsteps):
        if i % 10**2 == 0:
            print(i // 10 ** 2, "10^2 iterations")
        pid = np.random.randint(replicas)
        params = traj[pid, :, t[pid]]
        params_i = params
        if t[pid] % tau == 0:
            params_bar = traj[-1, :, i-1]
            params_i -= moving_rate * (params - params_bar)
            params_bar += moving_rate * (params - params_bar)
            traj[-1, :, i] = params_bar
        else:
            traj[-1, :, i] = traj[-1, :, i-1]
        x = np.random.choice(X, size=100)
        l1, l2 = E(x, params)
        assert not np.isnan(np.sum(l1))
        assert not np.isnan(np.sum(l2))
        params_i = M(x, l1, l2)
        t[pid] += 1
        traj[pid, :, t[pid]] = params_i
    return traj, t

# run simulations
X = np.hstack((d1_X, d2_X))
q = np.array([[0.5, 2, 4, 0.25, 5],
              [0.5, 2, 4, 0.25, 5]])

traj, t = run_simulation(q, X, replicas=2, Nsteps=5*10**2, tau=4, h=0.5, moving_rate=0.1)
#print(t)

# a1
for _ in range(traj.shape[0] - 1):
    #print(np.arange(t[_]), traj[_, :t[_]])
    plt.plot(traj[_, 0, :t[_]], alpha = 0.5, label="worker"+str(_))
central_traj = traj[-1, 0, np.arange(0, traj.shape[2], traj.shape[0] - 1)]
plt.plot(central_traj, label="master")
plt.legend()
plt.ylabel("a1")
plt.savefig("EM/a1.png")
plt.close()

# mu1
for _ in range(traj.shape[0] - 1):
    #print(np.arange(t[_]), traj[_, :t[_]])
    plt.plot(traj[_, 1, :t[_]], alpha = 0.5, label="worker"+str(_))
central_traj = traj[-1, 1, np.arange(0, traj.shape[2], traj.shape[0] - 1)]
plt.plot(central_traj, label="master")
plt.legend()
plt.ylabel("mu1")
plt.savefig("EM/mu1.png")
plt.close()

# mu2
for _ in range(traj.shape[0] - 1):
    #print(np.arange(t[_]), traj[_, :t[_]])
    plt.plot(traj[_, 2, :t[_]], alpha = 0.5, label="worker"+str(_))
central_traj = traj[-1, 2, np.arange(0, traj.shape[2], traj.shape[0] - 1)]
plt.plot(central_traj, label="master")
plt.legend()
plt.ylabel("mu2")
plt.savefig("EM/mu2.png")
plt.close()

# sigma1
for _ in range(traj.shape[0] - 1):
    #print(np.arange(t[_]), traj[_, :t[_]])
    plt.plot(traj[_, 3, :t[_]], alpha = 0.5, label="worker"+str(_))
central_traj = traj[-1, 3, np.arange(0, traj.shape[2], traj.shape[0] - 1)]
plt.plot(central_traj, label="master")
plt.legend()
plt.ylabel("sigma1")
plt.savefig("EM/sigma1.png")
plt.close()

# sigma2
for _ in range(traj.shape[0] - 1):
    #print(np.arange(t[_]), traj[_, :t[_]])
    plt.plot(traj[_, 4, :t[_]], alpha = 0.5, label="worker"+str(_))
central_traj = traj[-1, 4, np.arange(0, traj.shape[2], traj.shape[0] - 1)]
plt.plot(central_traj, label="master")
plt.legend()
plt.ylabel("sigma2")
plt.savefig("EM/sigma2.png")
plt.close()