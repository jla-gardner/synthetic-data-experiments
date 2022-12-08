import math
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
from experiments.data import descriptor_length
from experiments.saving import model_path

from experiments.util import test_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def tensor(x): return torch.FloatTensor(x).to(device)


_ACTIVATIONS = dict(
    celu=nn.CELU(), tanh=nn.Tanh(), relu=nn.ReLU()
)


def pairs(arr): return zip(arr, arr[1:])


class NN(nn.Module):
    def __init__(self, layers, activation, activate_last=False):
        super().__init__()

        activation = _ACTIVATIONS[activation]

        last_layer = nn.Linear(layers[-2], layers[-1])
        if activate_last:
            last_layer = nn.Sequential(last_layer, activation)

        other_layers = nn.Sequential(*[
            nn.Sequential(nn.Linear(a, b), activation)
            for a, b in pairs(layers[:-1])
        ])

        self._net = nn.Sequential(*other_layers, last_layer)

    def forward(self, x):
        return self._net(x)

    def save(self, path):
        torch.save(self.state_dict(),  path)

    def load(self, path, device):
        self.load_state_dict(torch.load(path, map_location=device))


def load_nn_model(**config):
    start = descriptor_length(config['n_max'], config['l_max'])
    layers = [start, *[config['width']] * config['depth'], 1]
    model = NN(layers, config['activation'])
    model.load(model_path('nn', config), device)
    return model


def train_step(loss_fn, data, opt):
    if hasattr(loss_fn, "train"):
        loss_fn.train()

    for d in data:
        opt.zero_grad()
        loss = loss_fn(*d)
        loss.backward()
        opt.step()


class Supervised(nn.Module):
    def __init__(self, model, criterion):
        super().__init__()
        self.model = model
        self.criterion = criterion

    def forward(self, x, y):
        out = self.model(x)
        return self.criterion(out, y)


def data_loader(x, y, batchsize=1024):
    return DataLoader(TensorDataset(x, y), batch_size=batchsize, shuffle=True)


def standardizer(a):
    scaler = StandardScaler()
    scaler.fit(a)
    return scaler.transform


def prepare_data(x_train, y_train, x_test, y_test):
    std = standardizer(x_train)
    x_train, x_test = tensor(std(x_train)), tensor(std(x_test))
    my = y_train.mean()
    y_train, y_test = tensor(y_train - my), tensor(y_test - my)
    return x_train, y_train, x_test, y_test


def train_nn_model(model, x, y, opt, scheduler, epochs, batchsize=4000, early_stopping=0):
    best_mae = math.inf
    mae_vals = []

    n = min(len(y) // 10, 1000)
    x, y, x_val, y_val = x[:-n], y[:-n], x[-n:], y[-n:]
    traindata = data_loader(x, y, batchsize)

    for epoch in range(epochs):
        model.train()

        train_step(Supervised(model, nn.MSELoss()).to(device), traindata, opt)
        scheduler.step()

        if (epoch + 1) % 5 == 0:
            model.eval()

            mae = test_model(model, x, y)['mae']
            mae_val = test_model(model, x_val, y_val)['mae']
            mae_vals.append(mae_val)

            print(f"{epoch+1:4d}", f"{mae*1000:>6.2f} meV",
                  f"{mae_val*1000:>6.2f} meV")

            if mae_val < best_mae:
                best_mae = mae_val
                model.save(".tmp.pt")

            # early stopping
            if early_stopping > 0 and best_mae not in mae_vals[-math.ceil(early_stopping / 5):]:
                print('stopping early')
                break

    model.load(".tmp.pt", device)
    return epoch + 1
