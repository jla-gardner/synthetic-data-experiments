
from time import perf_counter

import torch
from gpytorch.likelihoods import GaussianLikelihood
from torch.optim.lr_scheduler import ExponentialLR

from experiments import data_for, experiment, parse_passed_kwargs, test_model
from models.dkl import SVDKL, train_dkl_model
from models.nn import device, prepare_data


@experiment("dkl")
def main(*, fold, folds, n_train, n_max, l_max, train_on_equal_cns, **config):
    x_train, y_train, cn_train, x_test, y_test, cn_test = \
        data_for(fold, folds, n_max, l_max, n_train, train_on_equal_cns)
    x_train, y_train, x_test, y_test = \
        prepare_data(x_train, y_train, x_test, y_test)
    y_train, y_test = y_train.reshape(-1), y_test.reshape(-1)

    # model
    input_dim = x_train.shape[1]
    layers = [input_dim, *([config['width']] *
                           config['depth']), config['last_width']]
    model = SVDKL(layers, config['activation'],
                  config['m_sparse'], config['activate_last'])
    likelihood = GaussianLikelihood().to(device)

    # optimisers
    opt = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()},
    ], lr=config['lr'])
    scheduler = ExponentialLR(opt, config['decay'])

    tick = perf_counter()
    train_dkl_model(model, likelihood, x_train, y_train, opt, scheduler,
                    config['epochs'], batchsize=config['batchsize'], early_stopping=config['early_stopping'])
    tock = perf_counter()

    model.eval()
    likelihood.eval()

    results = {
        'train_duration': tock - tick,
        'testing': {
            'test_set': test_model(lambda x: model(x).mean, x_test, y_test, cn_test, batchsize=1000),
            'train_set': test_model(lambda x: model(x).mean, x_train, y_train, cn_train, batchsize=1000)
        }
    }

    return model, results


def run():
    defaults = dict(
        fold=0, folds=5,
        # training
        epochs=2_000, batchsize=860, decay=0.998, lr=0.000135, n_train=100,
        # model
        depth=2, width=200, last_width=6,
        activate_last=False, activation='celu',
        m_sparse=240,
        # other
        device=device.type, train_on_equal_cns=False,
        n_max=12, l_max=6, early_stopping=400
    )

    passed = parse_passed_kwargs(
        epochs=int, depth=int, width=int, last_width=int, decay=float,
        lr=float, n_train=int, activate_last=bool, m_sparse=int, fold=int,
        n_max=int, l_max=int, early_stopping=int, batchsize=int
    )

    config = {**{**defaults, **passed}}
    print(config)
    main(**config)
