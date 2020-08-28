import argparse
import copy
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


# configure settings for figures globally
fontsize = 64
plt.rcParams["figure.figsize"] = (40, 15)
plt.rcParams["lines.markersize"] = 10
plt.rcParams["axes.labelsize"] = fontsize
plt.rcParams["xtick.labelsize"] = fontsize
plt.rcParams["ytick.labelsize"] = fontsize
plt.rcParams['lines.linewidth'] = 5
plt.rcParams["legend.fontsize"] = fontsize
# plt.rcParams["grid.alpha"] = 0.75

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lin1 = nn.Linear(2, 100)
        #nn.init.kaiming_normal_(self.lin1.weight, mode="fan_out", nonlinearity="relu")
        self.lin2 = nn.Linear(100, 3)

    def forward(self, x):
        x = self.lin1(x)
        x = torch.relu(x)
        x = self.lin2(x)
        #output = torch.relu(x)
        #output = torch.sigmoid(x)
        output = F.log_softmax(x, dim=1)
        return output

def normalize(data):
    for i in range(data.shape[1]):
        mean = np.mean(data[:, i])
        var = np.var(data[:, i])
        data[:, i] = (data[:, i] - mean) / var
    return data

def train(model, device, train_loader, optimizer, epoch, log_interval=10):
    model.train()
    print("train loader\t", train_loader[0])
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            #print("output:\n", output)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            #print("pred:\n", pred)
            correct += pred.eq(target.view_as(pred)).sum().item()
            #print("target:\n", target.view_as(pred))

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def easgd(traindata, testdata, model, epochs=3, num_worker=2, batchsize=64,
           tau=10, learning_rate=0.05, moving_rate=0.05,
           log_interval=10, folder="spiral_easgd/",**kwargs):
    model().train()
    train_loader = torch.utils.data.DataLoader(
        traindata,
        batch_size=batchsize, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        testdata,
        batch_size=1000, shuffle=True, **kwargs)

    numbatch =  len(traindata) // batchsize
    models = [model()] * (num_worker+1)

    # To record train loss
    train_loss = np.ones((num_worker+1, len(traindata) // batchsize * epochs)) * 1.5
    masterloss = 1.5
    # current timestep for each worker
    t = np.zeros(num_worker, dtype=int)
    # train
    for epoch in range(1, epochs + 1):
        for batch_idx, (data, target) in enumerate(train_loader):
            # select a process
            pid = np.random.randint(num_worker)
            t[pid] += 1
            params = copy.deepcopy(models[pid])
            params_grad = copy.deepcopy(models[pid])
            params_i = models[pid]
            if t[pid] % tau == 0: # Now we communicate
                params_bar = models[-1]
                for p, p_i, p_bar in zip(params.parameters(), params_i.parameters(), params_bar.parameters()):
                    p_i.data = p_i.data - moving_rate * (p.data - p_bar.data)
                    p_bar.data = p_bar.data + moving_rate * (p.data - p_bar.data)
                masterloss = F.nll_loss(params_bar(data), target).item()

            optimizer = optim.SGD(params_grad.parameters(), lr=learning_rate)
            optimizer.zero_grad()
            loss = F.nll_loss(params_grad(data), target)
            loss.backward()
            optimizer.step()
            for p_i, g, p in zip(params_i.parameters(), params_grad.parameters(), params.parameters()):
                p_i.data += g.data - p.data

            if batch_idx % log_interval == 0:
                print('Train Epoch: {} of {}-th worker [{}/{} ({:.0f}%)]\tLoss: {:.6f} '.format(
                    epoch, pid, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))

            train_loss[:-1, (epoch - 1) * numbatch + batch_idx] = \
                train_loss[:-1, (epoch - 1) * numbatch + batch_idx - 1]
            train_loss[pid, (epoch - 1) * numbatch + batch_idx] = loss.item()
            train_loss[-1, (epoch - 1) * numbatch + batch_idx] = masterloss

    model().eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = models[-1](data)
            # print("output:\n", output)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            # print("pred:\n", pred)
            correct += pred.eq(target.view_as(pred)).sum().item()
            # print("target:\n", target.view_as(pred))

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    # plot
    for _ in range(train_loss.shape[0] - 1):
        plt.plot(train_loss[_, :], alpha=0.25, label="worker" + str(_))
    plt.plot(train_loss[-1, :], label="master")
    plt.legend(bbox_to_anchor=(1.0,  1.0))
    plt.ylabel("train loss")
    plt.xlabel("batch iteration")
    plt.ylim(0, 3)
    plt.savefig(folder + "bs" + str(batchsize) +
                "_tau" + str(tau) + ".png", bbox_inches='tight')
    plt.close()

    return train_loss[-1, :]


def measgd(traindata, testdata, model, epochs=3, num_worker=2, batchsize=64,
           tau=10, learning_rate=0.25, moving_rate=0.05, momentum_term = 0.25,
           log_interval=10, folder="spiral_measgd/",**kwargs):
    model().train()
    train_loader = torch.utils.data.DataLoader(
        traindata,
        batch_size=batchsize, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        testdata,
        batch_size=1000, shuffle=True, **kwargs)

    numbatch =  len(traindata) // batchsize
    models = [model()] * (num_worker+1)

    # To record train loss
    train_loss = np.ones((num_worker+1, len(traindata) // batchsize * epochs)) * 1.5
    masterloss = 1.5
    # current timestep for each worker
    t = np.zeros(num_worker, dtype=int)
    # Initialize Zero Momentum for weights and bias
    v = [None] * num_worker
    for i in range(num_worker):
        v[i] = model()
        for v_i in v[i].parameters():
            v_i.data.fill_(0.0)
    # train
    for epoch in range(1, epochs + 1):
        for batch_idx, (data, target) in enumerate(train_loader):
            # select a process
            pid = np.random.randint(num_worker)
            t[pid] += 1
            params = copy.deepcopy(models[pid])
            params_grad = copy.deepcopy(models[pid])
            params_i = models[pid]
            if t[pid] % tau == 0: # Now we communicate
                params_bar = models[-1]
                for p, p_i, p_bar in zip(params.parameters(), params_i.parameters(), params_bar.parameters()):
                    p_i.data = p_i.data - moving_rate * (p.data - p_bar.data)
                    p_bar.data = p_bar.data + moving_rate * (p.data - p_bar.data)
                masterloss = F.nll_loss(params_bar(data), target).item()

            optimizer = optim.SGD(params_grad.parameters(), lr=learning_rate)
            optimizer.zero_grad()
            loss = F.nll_loss(params_grad(data), target)
            loss.backward()
            optimizer.step()
            for v_i, g, p in zip(v[pid].parameters(), params_grad.parameters(), params.parameters()):
                v_i.data = momentum_term * v_i.data + (g.data - p.data)

            for v_i, p_i in zip(v[pid].parameters(), params_i.parameters()):
                p_i.data += v_i.data

            if batch_idx % log_interval == 0:
                print('Train Epoch: {} of {}-th worker [{}/{} ({:.0f}%)]\tLoss: {:.6f} '.format(
                    epoch, pid, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))

            train_loss[:-1, (epoch - 1) * numbatch + batch_idx] = \
                train_loss[:-1, (epoch - 1) * numbatch + batch_idx - 1]
            train_loss[pid, (epoch - 1) * numbatch + batch_idx] = loss.item()
            train_loss[-1, (epoch - 1) * numbatch + batch_idx] = masterloss

    model().eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = models[-1](data)
            # print("output:\n", output)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            # print("pred:\n", pred)
            correct += pred.eq(target.view_as(pred)).sum().item()
            # print("target:\n", target.view_as(pred))

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    # plot
    for _ in range(train_loss.shape[0] - 1):
        plt.plot(train_loss[_, :], alpha=0.25, label="worker" + str(_))
    plt.plot(train_loss[-1, :], label="master")
    plt.legend(bbox_to_anchor=(1.0,  1.0))
    plt.ylabel("train loss")
    plt.xlabel("batch iteration")
    plt.ylim(0, 3)
    plt.savefig(folder + "bs" + str(batchsize)  +
                "_tau" + str(tau) + "_del" + str(momentum_term) + ".png", bbox_inches='tight')
    plt.close()

    return train_loss[-1, :]




def easgld(traindata, testdata, model, epochs=3, num_worker=2, batchsize=64,
           tau=10, step_size=0.25, moving_rate=0.05, epsilon=1e-4, gamma=1.0,
           log_interval=10, folder="spiral_easgld/",**kwargs):
    model().train()
    train_loader = torch.utils.data.DataLoader(
        traindata,
        batch_size=batchsize, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        testdata,
        batch_size=1000, shuffle=True, **kwargs)

    numbatch =  len(traindata) // batchsize
    models = [model()] * (num_worker+1)

    # To record train loss
    train_loss = np.ones((num_worker+1, len(traindata) // batchsize * epochs)) * 1.5
    masterloss = 1.5
    # current timestep for each worker
    t = np.zeros(num_worker, dtype=int)
    # Initialize Zero Momentum for weights and bias
    v = [None] * num_worker
    for i in range(num_worker):
        v[i] = model()
        for v_i in v[i].parameters():
            v_i.data.fill_(0.0)
    # train
    for epoch in range(1, epochs + 1):
        for batch_idx, (data, target) in enumerate(train_loader):
            # select a process
            pid = np.random.randint(num_worker)
            t[pid] += 1
            params = copy.deepcopy(models[pid])
            params_grad = copy.deepcopy(models[pid])
            params_i = models[pid]
            if t[pid] % tau == 0: # Now we communicate
                params_bar = models[-1]
                for p, p_i, p_bar in zip(params.parameters(), params_i.parameters(), params_bar.parameters()):
                    p_i.data = p_i.data - moving_rate * (p.data - p_bar.data)
                    p_bar.data = p_bar.data + moving_rate * (p.data - p_bar.data)
                masterloss = F.nll_loss(params_bar(data), target).item()
            R = copy.deepcopy(params_i)
            for r in R.parameters():
                r.data.normal_()
            for v_i, r, p in zip(v[pid].parameters(), R.parameters(), params.parameters()):
                v_i.data = np.exp(-gamma * step_size) * v_i.data + np.sqrt(1 - np.exp(-2 * gamma * step_size * epsilon)) * r.data

            optimizer = optim.SGD(params_grad.parameters(), lr=step_size)
            optimizer.zero_grad()
            loss = F.nll_loss(params_grad(data), target)
            loss.backward()
            optimizer.step()
            for v_i, g, p in zip(v[pid].parameters(), params_grad.parameters(), params.parameters()):
                v_i.data += g.data - p.data

            for v_i, p_i in zip(v[pid].parameters(), params_i.parameters()):
                p_i.data += step_size * v_i.data

            if batch_idx % log_interval == 0:
                print('Train Epoch: {} of {}-th worker [{}/{} ({:.0f}%)]\tLoss: {:.6f} '.format(
                    epoch, pid, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))

            train_loss[:-1, (epoch - 1) * numbatch + batch_idx] = \
                train_loss[:-1, (epoch - 1) * numbatch + batch_idx - 1]
            train_loss[pid, (epoch - 1) * numbatch + batch_idx] = loss.item()
            train_loss[-1, (epoch - 1) * numbatch + batch_idx] = masterloss

    model().eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = models[-1](data)
            # print("output:\n", output)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            # print("pred:\n", pred)
            correct += pred.eq(target.view_as(pred)).sum().item()
            # print("target:\n", target.view_as(pred))

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    # plot
    for _ in range(train_loss.shape[0] - 1):
        plt.plot(train_loss[_, :], alpha=0.25, label="worker" + str(_))
    plt.plot(train_loss[-1, :], label="master")
    plt.legend(bbox_to_anchor=(1.0,  1.0))
    plt.ylabel("train loss")
    plt.xlabel("batch iteration")
    plt.ylim(0, 3)
    plt.savefig(folder + "bs" + str(batchsize) +
                "_tau" + str(tau) + "_eps" + str(epsilon) + ".png", bbox_inches='tight')
    plt.ylim()
    plt.close()

    return train_loss[-1, :]

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=200, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.15, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}


    # load data
    sample = np.load("data/sample.npy")
    label = np.load("data/label.npy")
    Xtest = np.load("data/Xtest.npy")
    Ytest = np.load("data/Ytest.npy")

    # to torch dataset
    X = torch.Tensor(sample)
    Y = torch.Tensor(label).long()

    traindata = torch.utils.data.TensorDataset(X, Y)
    testdata = torch.utils.data.TensorDataset(torch.Tensor(Xtest),
                                              torch.Tensor(Ytest).long())

    mv = 1e-7
    batchsize = [3, 10, 30]
    p = 4
    t = 50
    delta = [0.8, 0.4, 0.2, 0.1]
    eps = [1e-2, 1e-3, 1e-4, 1e-5]
    epochs = [15, 30, 50]
    for bs, epoch in zip(batchsize, epochs):
        print("EASGD with batch size", bs)
        easgd(traindata, testdata, model=Net, epochs=epoch, num_worker=p, batchsize=bs,
              tau=t, learning_rate=args.lr, moving_rate=mv,
              log_interval=args.log_interval,**kwargs)
        for d in delta:
            print("Momentum EASGD with batch size", bs, "and momentum term", d)
            measgd(traindata, testdata, model=Net, epochs=epoch, num_worker=p, batchsize=bs,
                   tau=t, learning_rate=args.lr, momentum_term=d, moving_rate=mv,
                   log_interval=args.log_interval, **kwargs)
        for e in eps:
            print("EASGLD with batch size", bs, "and perturbation term", e)
            easgld(traindata, testdata, model=Net, epochs=epoch, num_worker=p, batchsize=bs, tau=t,
                   step_size=args.lr, moving_rate=mv, epsilon=e, log_interval=args.log_interval,**kwargs)


if __name__ == '__main__':
    main()