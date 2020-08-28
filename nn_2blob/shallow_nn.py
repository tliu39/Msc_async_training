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
plt.rcParams["figure.figsize"] = (25, 20)
plt.rcParams["lines.markersize"] = 10
plt.rcParams["axes.labelsize"] = fontsize
plt.rcParams["xtick.labelsize"] = fontsize
plt.rcParams["ytick.labelsize"] = fontsize
plt.rcParams['lines.linewidth'] = 10
plt.rcParams["legend.fontsize"] = fontsize
#plt.rcParams["grid.alpha"] = 0.75

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lin1 = nn.Linear(2, 8)
        #nn.init.kaiming_normal_(self.lin1.weight, mode="fan_out", nonlinearity="relu")
        self.lin2 = nn.Linear(8, 2)

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

def test(model, test_loader, train=False):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            #print("output:\n", output)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            #print("pred:\n", pred)
            correct += pred.eq(target.view_as(pred)).sum().item()
            #print("target:\n", target.view_as(pred))

    test_loss /= len(test_loader.dataset)
    if train:
        print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    else:
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    return len(test_loader.dataset) - correct


# def easgld(traindata, testdata, model, epochs=3, num_worker=2, batchsize=64,
#            tau=10, step_size=0.25, moving_rate=0.05, gamma=1.0, epsilon=1e-4,
#            log_interval=10, folder = "nn2blob_easgld/",**kwargs):
#     model().train()
#     train_loader = torch.utils.data.DataLoader(
#         traindata,
#         batch_size=batchsize, shuffle=True, **kwargs)
#
#     test_loader = torch.utils.data.DataLoader(
#         testdata,
#         batch_size=1000, shuffle=True, **kwargs)
#
#     numbatch =  len(traindata) // batchsize
#     models = [model()] * (num_worker+1)
#
#     # To record train loss
#     train_loss = np.ones((num_worker+1, len(traindata) // batchsize * epochs))
#     masterloss = 1.0
#     # current timestep for each worker
#     t = np.zeros(num_worker, dtype=int)
#     # Initialize Zero Momentum for weights and bias
#     v = [None] * num_worker
#     for i in range(num_worker):
#         v[i] = model()
#         for v_i in v[i].parameters():
#             v_i.data.fill_(0.0)
#     # train
#     for epoch in range(1, epochs + 1):
#         for batch_idx, (data, target) in enumerate(train_loader):
#             # select a process
#             pid = np.random.randint(num_worker)
#             t[pid] += 1
#             params = copy.deepcopy(models[pid])
#             params_grad = copy.deepcopy(models[pid])
#             params_i = models[pid]
#             if t[pid] % tau == 0: # Now we communicate
#                 params_bar = models[-1]
#                 for p, p_i, p_bar in zip(params.parameters(), params_i.parameters(), params_bar.parameters()):
#                     p_i.data = p_i.data - moving_rate * (p.data - p_bar.data)
#                     p_bar.data = p_bar.data + moving_rate * (p.data - p_bar.data)
#                 masterloss = F.nll_loss(params_bar(data), target).item()
#             R = copy.deepcopy(params_i)
#             for r in R.parameters():
#                 r.data.normal_()
#             for v_i, r, p in zip(v[pid].parameters(), R.parameters(), params.parameters()):
#                 v_i.data = np.exp(-gamma * step_size) * v_i.data + np.sqrt(1 - np.exp(-2 * gamma * step_size * epsilon)) * r.data
#
#             optimizer = optim.SGD(params_grad.parameters(), lr=step_size)
#             optimizer.zero_grad()
#             loss = F.nll_loss(params_grad(data), target)
#             loss.backward()
#             optimizer.step()
#             for v_i, g, p in zip(v[pid].parameters(), params_grad.parameters(), params.parameters()):
#                 v_i.data += g.data - p.data
#
#             for v_i, p_i in zip(v[pid].parameters(), params_i.parameters()):
#                 p_i.data += step_size * v_i.data
#
#             if batch_idx % log_interval == 0:
#                 print('Train Epoch: {} of {}-th worker [{}/{} ({:.0f}%)]\tLoss: {:.6f} '.format(
#                     epoch, pid, batch_idx * len(data), len(train_loader.dataset),
#                            100. * batch_idx / len(train_loader), loss.item()))
#
#             train_loss[:-1, (epoch - 1) * numbatch + batch_idx] = \
#                 train_loss[:-1, (epoch - 1) * numbatch + batch_idx - 1]
#             train_loss[pid, (epoch - 1) * numbatch + batch_idx] = loss.item()
#             train_loss[-1, (epoch - 1) * numbatch + batch_idx] = masterloss
#
#     model().eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             output = models[-1](data)
#             # print("output:\n", output)
#             test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
#             pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
#             # print("pred:\n", pred)
#             correct += pred.eq(target.view_as(pred)).sum().item()
#             # print("target:\n", target.view_as(pred))
#
#     test_loss /= len(test_loader.dataset)
#
#     print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         test_loss, correct, len(test_loader.dataset),
#         100. * correct / len(test_loader.dataset)))
#
#     # plot
#     for _ in range(train_loss.shape[0] - 1):
#         plt.plot(train_loss[_, :], alpha=0.25, label="worker" + str(_))
#     plt.plot(train_loss[-1, :], alpha=0.75, label="master")
#     plt.legend()
#     plt.ylabel("train loss")
#     plt.xlabel("batch iteration")
#     plt.savefig(folder + "p" + str(num_worker) + "_tau" + str(tau) + "_eps" + str(epsilon) + ".png")
#     plt.close()

def easgld(traindata, testdata, model, epochs=3, num_worker=2, batchsize=64,
           tau=10, step_size=0.25, moving_rate=0.05, gamma=1.0, epsilon=1e-4,
           log_interval=10, folder="spiral_easgld/",**kwargs):
    model().train()
    train_loader = torch.utils.data.DataLoader(
        traindata,
        batch_size=batchsize, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        testdata,
        batch_size=1000, shuffle=True, **kwargs)

    numbatch = len(traindata) // batchsize
    models = [model()] * (num_worker+1)

    # To record train loss
    train_loss = np.ones((num_worker+1, numbatch * epochs)) * 1.5
    train_epoch_loss = np.zeros((num_worker+1, numbatch * epochs + 1))
    test_loss = np.zeros(epochs+1)

    for i in range(num_worker+1):
        train_epoch_loss[i, 0] = test(models[i], train_loader, train=True)
    test_loss[0] = test(models[-1], test_loader)

    masterloss = 1.5
    # current timestep for each worker
    t = np.zeros(num_worker, dtype=int)
    # Initialize Zero Momentum for weights and bias
    v = [model()] * num_worker
    for i in range(num_worker):
        for v_i in v[i].parameters():
            v_i.data.fill_(0.0)
    # train
    for epoch in range(epochs):
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

            # if batch_idx % log_interval == 0:
            #     print('Train Epoch: {} of {}-th worker [{}/{} ({:.0f}%)]\tLoss: {:.6f} '.format(
            #         epoch + 1, pid, batch_idx * len(data), len(train_loader.dataset),
            #                100. * batch_idx / len(train_loader), loss.item()))


            train_loss[:-1, epoch * numbatch + batch_idx] = train_loss[:-1, epoch * numbatch + batch_idx - 1]

            train_loss[pid, epoch * numbatch + batch_idx] = loss.item()
            train_loss[-1, epoch * numbatch + batch_idx] = masterloss

            train_epoch_loss[:, epoch * numbatch + batch_idx + 1] = train_epoch_loss[:, epoch * numbatch + batch_idx]
            train_epoch_loss[pid, epoch * numbatch + batch_idx + 1] = test(models[pid], train_loader, train=True)
            if t[pid] % tau == 0:
                train_epoch_loss[-1, epoch * numbatch + batch_idx + 1] = test(models[-1], train_loader, train=True)
            test_loss[epoch + 1] = test(models[-1], test_loader)
    return train_loss, train_epoch_loss, test_loss





def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}


    # load data
    npdata = np.load("data.npy")
    nptest = np.load("test.npy")

    # normalize data
    # scaler = StandardScaler()
    # npdata[:, 0:2] = scaler.fit_transform(npdata[:, 0:2])
    # nptest[:, 0:2] = scaler.transform(nptest[:, 0:2])

    # plot data
    plt.scatter(npdata[np.where(npdata[:, 2] == 0), 0],npdata[np.where(npdata[:, 2] == 0), 1], label="cluster 0")
    plt.scatter(npdata[np.where(npdata[:, 2] == 1), 0],npdata[np.where(npdata[:, 2] == 1), 1], label="cluster 1")
    plt.legend()
    plt.savefig("visualization/normal_data.png")
    plt.close()

    plt.scatter(npdata[np.where(npdata[:, 2] == 0), 0],npdata[np.where(npdata[:, 2] == 0), 1], label="cluster 0")
    plt.scatter(npdata[np.where(npdata[:, 2] == 1), 0],npdata[np.where(npdata[:, 2] == 1), 1], label="cluster 1")
    plt.legend()
    plt.show()
    plt.close()

    # split samples and labels
    Xtrain = torch.Tensor(npdata[:, 0:2])
    Ytrain = torch.Tensor(npdata[:, 2]).long()
    Xtest = torch.Tensor(nptest[:, 0:2])
    Ytest = torch.Tensor(nptest[:, 2]).long()

    traindata = torch.utils.data.TensorDataset(Xtrain, Ytrain)
    testdata = torch.utils.data.TensorDataset(Xtest, Ytest)

    ps = [2]
    taus = [4]#[1, 2, 4]
    eps = [1e-2]#[1e-2, 1e-3, 1e-4]
    for p, t, e in product(ps, taus, eps):
        print(p, t, e)
        train_loss, train_epoch_loss, test_loss = easgld(traindata, testdata, model=Net, epochs=2+2*p, num_worker=p,
                                             batchsize=8, tau=t, step_size=0.05, moving_rate=1e-6,
                                             gamma=1.0, epsilon=e, log_interval=args.log_interval,**kwargs)
        for i in range(train_epoch_loss.shape[0] - 1):
            plt.plot(train_epoch_loss[i, :], alpha=0.5, label="worker" + str(i))
        plt.plot(train_epoch_loss[-1, :], label="master")
        plt.legend()
        plt.ylabel("training error")
        plt.xlabel("batch iteration")
        plt.ylim(bottom=0)
        plt.savefig("easgld/" + "p" + str(p) + "_tau" + str(t) + "_eps" + str(e) + ".png")
        plt.close()
        print(p, t, e, "\n"*5)

if __name__ == '__main__':
    main()