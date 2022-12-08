from time import perf_counter

import torch

from torch.optim.lr_scheduler import ExponentialLR

from experiments import data_for, experiment, parse_passed_kwargs, test_model
from models.nn import NN, prepare_data, device, train_nn_model


@experiment("nn")
def main(*, fold, folds, n_train, n_max, l_max, train_on_equal_cns, **config):
    x_train, y_train, cn_train, x_test, y_test, cn_test = \
        data_for(fold, folds, n_max, l_max, n_train, train_on_equal_cns)

    x_train, y_train, x_test, y_test = \
        prepare_data(x_train, y_train, x_test, y_test)

    layers = [x_train.shape[1], *([config['width']] * config['depth']), 1]
    model = NN(layers, config['activation']).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = ExponentialLR(opt, config['decay'])

    tick = perf_counter()
    actual_epochs = train_nn_model(model, x_train, y_train, opt, scheduler,
                                   config['epochs'], config['batchsize'], config['early_stopping'])
    tock = perf_counter()

    results = {
        'train_duration': tock - tick,
        'actual_epochs': actual_epochs,
        'testing': {
            'test_set': test_model(model, x_test, y_test, cn_test),
            'train_set': test_model(model, x_train, y_train, cn_train)
        }
    }

    return model, results


def run():
    defaults = dict(
        folds=5, n_train=100,
        epochs=1000, depth=4, width=330,
        device_type=device.type, decay=0.9985,
        lr=0.0003, activation="celu", batchsize=1000,
        n_max=12, l_max=6,
        train_on_equal_cns=False, early_stopping=50
    )

    passed = parse_passed_kwargs(
        epochs=int, depth=int, width=int, decay=float, lr=float, n_train=int,
        n_max=int, l_max=int, early_stopping=int, batchsize=int, train_on_equal_cns=bool
    )

    config = {**{**defaults, **passed}}
    print(config)
    for fold in range(5):
        main(fold=fold, **config)
