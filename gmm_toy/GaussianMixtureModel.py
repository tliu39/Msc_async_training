import numpy as np


class GaussianMixureModel:
    def __init__(self, num_workers=2, same_target_dist=True, datafile1="data/data1.npy", datafile2="data/data2.npy"):
        d1 = np.load(datafile1)
        d2 = np.load(datafile2)
        self.num_workers = num_workers
        if same_target_dist:
            self.X1 = np.hstack((d1[:, 0], d2[:, 0]))
            self.X2 = np.hstack((d1[:, 0], d2[:, 0]))
        else:
            self.X1 = d1[:, 0]
            self.X2 = d2[:, 0]
        self.X = np.hstack((self.X1, self.X2))

    def sgd(self, x, params):
        """Stochastic Gradient Descent"""
        a1, mu1, mu2, sigma1, sigma2 = params

        # the likelihood for observing samples from class 1
        l1 = a1/(sigma1 * np.sqrt(2 * np.pi)) * np.exp(-1/2 * ((x - mu1) / sigma1)**2)
        # the likelihood for observing samples from class 2
        l2 = (1-a1)/(sigma2 * np.sqrt(2 * np.pi)) * np.exp(-1/2 * ((x - mu2) / sigma2)**2)
        prob1 = l1 / (l1 + l2)      # the probability that samples are from class 1
        prob2 = l2 / (l1 + l2)      # the probability that samples are from class 2

        # derivatives
        dEdmu1 = np.mean(prob1 * (x - mu1) / sigma1**2)
        dEdmu2 = np.mean(prob2 * (x - mu2) / sigma2**2)
        dEdsigma1 = np.mean(prob1 * ((x - mu1)**2 / sigma1**2 - 1))
        dEdsigma2 = np.mean(prob2 * ((x - mu2)**2 / sigma2**2 - 1))

        return -1 * np.array([0, dEdmu1, dEdmu2, dEdsigma1, dEdsigma2])

    def easgd_train(self, q, gradient_function, num_iterations=10 ** 4, tau=10, learning_rate=0.01, moving_rate=0.01, batch_size=16):
        """define asynchronous easgd training processes"""
        traj = np.empty((self.num_workers + 1, 5, num_iterations))
        traj[:, :, :] = np.nan
        traj[:self.num_workers, :, 0] = np.copy(q)
        traj[self.num_workers, :, 0] = np.mean(q, axis=0)
        t = np.zeros(self.num_workers, dtype=int)

        for i in range(1, num_iterations):
            traj[:-1, :, i] = traj[:-1, :, i - 1]
            if i % 10 ** 4 == 0:
                print(i // 10 ** 4, "10^4 iterations")
            pid = np.random.randint(self.num_workers)
            params = traj[pid, :, i-1]
            params_i = np.copy(params)
            if t[pid] % tau == 0:
                params_bar = traj[-1, :, i - 1]
                params_i -= moving_rate * (params - params_bar)
                params_bar += moving_rate * (params - params_bar)
                traj[-1, :, i] = params_bar
            else:
                traj[-1, :, i] = traj[-1, :, i - 1]
            if pid == 0:
                x = np.random.choice(self.X1, size=batch_size)
            elif pid == 1:
                x = np.random.choice(self.X2, size=batch_size)
            else:
                x = np.random.choice(self.X, size=batch_size)
            params_i -= learning_rate*gradient_function(self, x, params)
            t[pid] += 1
            traj[pid, :, i] = params_i
        return traj, t

    def easgld_train(self, q, gradient_function, num_iterations=10 ** 4, tau=10, step_size=0.25, moving_rate=0.05, gamma=1.0, epsilon=1e-4, batch_size=16):
        """define asynchronous easgld training processes with friction term gamma and perturbation term epsilon"""
        traj = np.empty((self.num_workers + 1, 5, num_iterations))
        traj[:, :, :] = np.nan
        traj[:self.num_workers, :, 0] = np.copy(q)
        traj[self.num_workers, :, 0] = np.mean(q, axis=0)
        t = np.zeros(self.num_workers, dtype=int)
        v = np.zeros((self.num_workers, 5))

        for i in range(1, num_iterations):
            traj[:-1, :, i] = traj[:-1, :, i-1]
            if i % 10 ** 4 == 0:
                print(i // 10 ** 4, "10^4 iterations")
            pid = np.random.randint(self.num_workers)
            params = traj[pid, :, i-1]
            params_i = np.copy(params)
            if t[pid] % tau == 0:
                params_bar = traj[-1, :, i - 1]
                params_i -= moving_rate * (params - params_bar)
                params_bar += moving_rate * (params - params_bar)
                traj[-1, :, i] = params_bar
            else:
                traj[-1, :, i] = traj[-1, :, i - 1]
            if pid == 0:
                x = np.random.choice(self.X1, size=batch_size)
            elif pid == 1:
                x = np.random.choice(self.X2, size=batch_size)
            else:
                x = np.random.choice(self.X, size=batch_size)
            R = np.random.randn(params_i.size).reshape(params_i.shape)
            R[0] = 0
            v[pid, :] = np.exp(-gamma * step_size) * v[pid, :] + np.sqrt(1 - np.exp(-2 * gamma * step_size * epsilon)) * R
            v[pid, :] -= step_size * gradient_function(self, x, params)
            params_i += step_size * v[pid, :]
            t[pid] += 1
            traj[pid, :, i] = np.copy(params_i)
        return traj, t
