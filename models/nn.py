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


def pairs(arr): 
    """
    Return pairs of consecutive elements in an array
    """
    return zip(arr, arr[1:])


class NN(nn.Module):
    """
    Neural network model
    
    Parameters
    ----------
    
    layers : list of int
        Number of nodes in each layer
    activation : str
        Activation function
    activate_last : bool
        Whether to apply activation function to last layer
    """
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
    """
    Load a neural network model mataching config from disk
    """
    start = descriptor_length(config['n_max'], config['l_max'])
    layers = [start, *[config['width']] * config['depth'], 1]
    model = NN(layers, config['activation'])
    model.load(model_path('nn', config), device)
    return model


def train_step(loss_fn, data, opt):
    """
    perform a single training step

    Parameters
    ----------

    loss_fn : callable
        function that takes a batch of data and returns a loss
    data : iterable
        iterable of batches of data
    opt : torch.optim.Optimizer
        optimizer to use
    """
    if hasattr(loss_fn, "train"):
        loss_fn.train()

    for d in data:
        opt.zero_grad()
        loss = loss_fn(*d)
        loss.backward()
        opt.step()


class Supervised(nn.Module):
    """
    A supervised model
    
    Parameters
    ----------

    model : nn.Module
        model to train
    criterion : nn.Module
        loss function
    """

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
    """
    Return a function that standardizes a numpy array,
    using the mean and standard deviation of a
    """

    scaler = StandardScaler()
    scaler.fit(a)
    return scaler.transform


def prepare_data(x_train, y_train, x_test, y_test):
    """
    Prepare data for training by standardizing and converting to tensors
    
    Parameters
    ----------
    
    x_train : np.ndarray
        training data
    y_train : np.ndarray
        training labels
    x_test : np.ndarray
        test data
    y_test : np.ndarray
        test labels
    """
    std = standardizer(x_train)
    x_train, x_test = tensor(std(x_train)), tensor(std(x_test))
    my = y_train.mean()
    y_train, y_test = tensor(y_train - my), tensor(y_test - my)
    return x_train, y_train, x_test, y_test


def train_nn_model(model, x, y, opt, scheduler, epochs, batchsize=4000, early_stopping=0):
    """
    Train a neural network model
    
    Parameters
    ----------
    
    model : nn.Module
        model to train
    x : torch.Tensor
        training data
    y : torch.Tensor
        training labels
    opt : torch.optim.Optimizer
        optimizer to use
    scheduler : torch.optim.lr_scheduler
        learning rate scheduler
    epochs : int
        (max) number of epochs to train for
    batchsize : int
        batch size
    early_stopping : int
        number of epochs to wait before stopping if validation loss 
        does not improve. If 0, no early stopping is performed.
    """
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
