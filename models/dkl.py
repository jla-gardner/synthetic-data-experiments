import math
import torch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import VariationalELBO

from models.nn import NN, data_loader, device, Supervised, train_step
from experiments.util import test_model
from experiments.data import descriptor_length
from experiments.saving import model_path


class SVDKL(ApproximateGP):
    def __init__(self, layers, activation, m_sparse, activate_last=False):
        inducing_points = torch.rand(m_sparse, layers[0])
        if activation in ['tanh']:
            inducing_points = inducing_points * 2 - 1

        variational_distribution = CholeskyVariationalDistribution(m_sparse)
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution,
            learn_inducing_locations=True
        )

        super().__init__(variational_strategy)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())

        self.net = NN(layers, activation, activate_last)

    def forward(self, x):
        x = self.net(x)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def save(self, path):
        torch.save(self.state_dict(),  path)

    def load(self, path, device):
        self.load_state_dict(torch.load(path, map_location=device))


def load_dkl_model(**config):
    start = descriptor_length(config['n_max'], config['l_max'])
    layers = [start, *([config['width']] *
                       config['depth']), config['last_width']]
    model = SVDKL(layers, config['activation'],
                  config['m_sparse'], config['activate_last'])
    model.load(model_path('dkl', config), device)
    return model


def train_dkl_model(model, likelihood, x, y, opt, scheduler, epochs, batchsize=4000, early_stopping=0):
    mll = VariationalELBO(likelihood, model, num_data=len(y))
    def loss(a, b): return -mll(a, b)

    best_mae = math.inf
    mae_vals = []

    n = min(len(y) // 10, 1000)
    x, y, x_val, y_val = x[:-n], y[:-n], x[-n:], y[-n:]
    traindata = data_loader(x, y, batchsize)

    for epoch in range(epochs):
        model.train()
        likelihood.train

        train_step(Supervised(model, loss).to(device), traindata, opt)
        scheduler.step()

        if (epoch + 1) % 5 == 0:
            model.eval()
            likelihood.eval()

            mae = test_model(lambda x: model(x).mean, x,
                             y, batchsize=1000)['mae']
            mae_val = test_model(lambda x: model(x).mean, x_val,
                                 y_val, batchsize=1000)['mae']
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
    return model
