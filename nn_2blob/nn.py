import argparse
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
        self.lin1 = nn.Linear(2, 8)
        nn.init.kaiming_normal_(self.lin1.weight, mode="fan_out", nonlinearity="relu")
        self.lin2 = nn.Linear(8, 2)
        nn.init.kaiming_normal_(self.lin2.weight, mode="fan_out", nonlinearity="relu")
        self.lin3 = nn.Linear(2, 1)

    def forward(self, x):
        x = self.lin1(x)
        x = torch.relu(x)
        x = self.lin2(x)
        x = F.relu(x)
        x = self.lin3(x)
        output = torch.sigmoid(x)
        #output = F.log_softmax(x, dim=1)
        return output

def normalize(data):
    for i in range(data.shape[1]):
        mean = np.mean(data[:, i])
        var = np.var(data[:, i])
        data[:, i] = (data[:, i] - mean) / var
    return data

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.binary_cross_entropy_with_logits(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.25, metavar='LR',
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
    scaler = StandardScaler()
    npdata[:, 0:2] = scaler.fit_transform(npdata[:, 0:2])
    nptest[:, 0:2] = scaler.transform(nptest[:, 0:2])

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
    Ytrain = torch.Tensor(npdata[:, 2]).reshape(-1, 1)
    Xtest = torch.Tensor(nptest[:, 0:2])
    Ytest = torch.Tensor(nptest[:, 2]).reshape(-1, 1)

    torchdata = torch.utils.data.TensorDataset(Xtrain, Ytrain)
    train_loader = torch.utils.data.DataLoader(
        torchdata,
        batch_size=args.batch_size, shuffle=True, **kwargs)

    torchtest = torch.utils.data.TensorDataset(Xtest, Ytest)
    test_loader = torch.utils.data.DataLoader(
        torchtest,
        batch_size=args.batch_size, shuffle=True, **kwargs
    )


    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        scheduler.step()

    # if args.save_model:
    #     torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()