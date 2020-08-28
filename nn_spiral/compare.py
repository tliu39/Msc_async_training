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

def test(model, test_loader, train=False):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            #print("output:\n", output)
            #test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            #print("pred:\n", pred)
            correct += pred.eq(target.view_as(pred)).sum().item()
            #print("target:\n", target.view_as(pred))

    test_loss /= len(test_loader.dataset)
    # if train:
    #     print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
    #         test_loss, correct, len(test_loader.dataset),
    #         100. * correct / len(test_loader.dataset)))
    # else:
    #     print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
    #         test_loss, correct, len(test_loader.dataset),
    #         100. * correct / len(test_loader.dataset)))
    return len(test_loader.dataset) - correct

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

    numbatch = len(traindata) // batchsize
    models = [model()] * (num_worker+1)

    # To record train loss
    train_loss = np.ones((num_worker+1, numbatch * epochs)) * 1.5
    train_epoch_loss = np.zeros(epochs+1)
    test_loss = np.zeros(epochs+1)

    train_epoch_loss[0] = test(models[-1], train_loader, train=True)
    test_loss[0] = test(models[-1], test_loader)

    masterloss = 1.5
    # current timestep for each worker
    t = np.zeros(num_worker, dtype=int)
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

            optimizer = optim.SGD(params_grad.parameters(), lr=learning_rate)
            optimizer.zero_grad()
            loss = F.nll_loss(params_grad(data), target)
            loss.backward()
            optimizer.step()
            for p_i, g, p in zip(params_i.parameters(), params_grad.parameters(), params.parameters()):
                p_i.data += g.data - p.data
            # if batch_idx % log_interval == 0:
            #     print('Train Epoch: {} of {}-th worker [{}/{} ({:.0f}%)]\tLoss: {:.6f} '.format(
            #         epoch + 1, pid, batch_idx * len(data), len(train_loader.dataset),
            #                100. * batch_idx / len(train_loader), loss.item()))


            train_loss[:-1, epoch * numbatch + batch_idx] = train_loss[:-1, epoch * numbatch + batch_idx - 1]


            train_loss[pid, epoch * numbatch + batch_idx] = loss.item()
            train_loss[-1, epoch * numbatch + batch_idx] = masterloss

        train_epoch_loss[epoch + 1] = test(models[-1], train_loader, train=True)
        test_loss[epoch + 1] = test(models[-1], test_loader)
        if train_epoch_loss[epoch + 1] == 0:
            break
    return train_loss[-1, :], train_epoch_loss, test_loss


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

    numbatch = len(traindata) // batchsize
    models = [model()] * (num_worker+1)

    # To record train loss
    train_loss = np.ones((num_worker+1, numbatch * epochs)) * 1.5
    train_epoch_loss = np.zeros(epochs+1)
    test_loss = np.zeros(epochs+1)

    train_epoch_loss[0] = test(models[-1], train_loader, train=True)
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

            optimizer = optim.SGD(params_grad.parameters(), lr=learning_rate)
            optimizer.zero_grad()
            loss = F.nll_loss(params_grad(data), target)
            loss.backward()
            optimizer.step()
            for v_i, g, p in zip(v[pid].parameters(), params_grad.parameters(), params.parameters()):
                v_i.data = momentum_term * v_i.data + (g.data - p.data)

            for v_i, p_i in zip(v[pid].parameters(), params_i.parameters()):
                p_i.data += v_i.data

            # if batch_idx % log_interval == 0:
            #     print('Train Epoch: {} of {}-th worker [{}/{} ({:.0f}%)]\tLoss: {:.6f} '.format(
            #         epoch + 1, pid, batch_idx * len(data), len(train_loader.dataset),
            #                100. * batch_idx / len(train_loader), loss.item()))


            train_loss[:-1, epoch * numbatch + batch_idx] = train_loss[:-1, epoch * numbatch + batch_idx - 1]


            train_loss[pid, epoch * numbatch + batch_idx] = loss.item()
            train_loss[-1, epoch * numbatch + batch_idx] = masterloss

        train_epoch_loss[epoch + 1] = test(models[-1], train_loader, train=True)
        test_loss[epoch + 1] = test(models[-1], test_loader)
        if train_epoch_loss[epoch + 1] == 0:
            break
    return train_loss[-1, :], train_epoch_loss, test_loss




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
    train_epoch_loss = np.zeros(epochs+1)
    test_loss = np.zeros(epochs+1)

    train_epoch_loss[0] = test(models[-1], train_loader, train=True)
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

        train_epoch_loss[epoch + 1] = test(models[-1], train_loader, train=True)
        test_loss[epoch + 1] = test(models[-1], test_loader)
        if train_epoch_loss[epoch + 1] == 0:
            break
    return train_loss[-1, :], train_epoch_loss, test_loss




def main():
    # configure settings for figures globally
    fontsize = 64
    plt.rcParams["figure.figsize"] = (30, 20)
    plt.rcParams["lines.markersize"] = 10
    plt.rcParams["axes.labelsize"] = fontsize
    plt.rcParams["xtick.labelsize"] = fontsize
    plt.rcParams["ytick.labelsize"] = fontsize
    plt.rcParams['lines.linewidth'] = 5
    plt.rcParams["lines.marker"] = "D"
    plt.rcParams["lines.markersize"] = 15
    plt.rcParams["legend.fontsize"] = fontsize


    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=200, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.25, metavar='LR',
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

    mv = 1e-8
    batchsize = [300]#[3, 10]#, 10]#[3, 10, 30, 60, 90]
    p = 4
    tau = [25, 100, 400]#[25, 100, 400]
    delta = 0.5
    eps = [1e-2, 1e-4, 1e-6, 1e-8]
    epochs = [1000]#[200, 300]#, 100]#, 50]#[50, 50, 150, 150, 150]
    sims = 3

    for (bs, epoch) in zip(batchsize, epochs):
        for t in tau:
            step = epoch // 50
            indices = np.arange(0, epoch + 1, step)
            tick_indices = np.arange(0, indices.shape[0], 10)
            tick_labels = indices[tick_indices]
            train_epoch_loss = []
            test_loss = []
            print("\n\nNEW Chart", bs, t)
            print("EASGD with batch size", bs)
            train = np.zeros((sims, epoch+1))
            test = np.zeros((sims, epoch+1))
            for sim in range(sims):
                _, train[sim, :], test[sim, :] = \
                    easgd(traindata, testdata, model=Net, epochs=epoch, num_worker=p,
                          batchsize=bs, tau=t, learning_rate=args.lr, moving_rate=mv,
                          log_interval=args.log_interval, **kwargs)
            train_epoch_loss.append(np.copy(np.mean(train, axis=0)))
            test_loss.append(np.copy(np.mean(test, axis=0)))

            print("\nMomentum EASGD with batch size", bs, "and momentum term", delta)
            for sim in range(sims):
                _, train[sim, :], test[sim, :] = \
                    measgd(traindata, testdata, model=Net, epochs=epoch, num_worker=p,
                           batchsize=bs, tau=t, learning_rate=args.lr, momentum_term=delta,
                           moving_rate=mv, log_interval=args.log_interval, **kwargs)
            train_epoch_loss.append(np.copy(np.mean(train, axis=0)))
            test_loss.append(np.copy(np.mean(test, axis=0)))

            for i, e in enumerate(eps):
                print("\nEASGLD with batch size", bs, "and perturbation term", e)
                for sim in range(sims):
                    _, train[sim, :], test[sim, :] = \
                        easgld(traindata, testdata, model=Net, epochs=epoch, num_worker=p,
                               batchsize=bs, tau=t, step_size=args.lr, moving_rate=mv, gamma=1.0,
                               epsilon=e, log_interval=args.log_interval, **kwargs)
                train_epoch_loss.append(np.copy(np.mean(train, axis=0)))
                test_loss.append(np.copy(np.mean(test, axis=0)))

            # plot train epoch
            plt.plot(train_epoch_loss[0][indices], label="EASGD")
            plt.plot(train_epoch_loss[1][indices], label="MEASGD with $\delta =$" + str(delta))
            for i, e in enumerate(eps):
                plt.plot(train_epoch_loss[i + 2][indices], label="EASGLD $\epsilon =$" + str(e))

            # plt.ylim(0, 1.0)
            plt.legend()
            plt.xticks(tick_indices, tick_labels)
            plt.xlabel("epoch")
            plt.ylabel("averaged train error")
            #plt.yscale('log')
            plt.savefig("compare_train_epoch/bs" + str(bs) + "_tau" + str(t) + ".png", bbox_inches='tight')
            plt.close()

            # plot test
            plt.plot(test_loss[0][indices], label="EASGD")
            plt.plot(test_loss[1][indices], label="MEASGD with $\delta =$" + str(delta))
            for i, e in enumerate(eps):
                plt.plot(test_loss[i + 2][indices], label="EASGLD $\epsilon =$" + str(e))
            # plt.ylim(0, 1.0)
            plt.legend()
            plt.xticks(tick_indices, tick_labels)
            plt.xlabel("epoch")
            plt.ylabel("averaged test error")
            #plt.yscale('log')
            plt.savefig("compare_test/bs" + str(bs) + "_tau" + str(t) + ".png", bbox_inches='tight')
            plt.close()

if __name__ == '__main__':
    main()