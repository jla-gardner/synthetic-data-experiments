from time import perf_counter as now

from experiments import data_for, experiment, test_model, parse_passed_kwargs
from models.gpr import SparseGPR


@experiment("gpr")
def main(*, fold, folds, m_sparse, n_train, n_max, l_max, noise, train_on_equal_cns):

    x_train, y_train, cn_train, x_test, y_test, cn_test = \
        data_for(fold, folds, n_max, l_max, n_train, train_on_equal_cns)

    model = SparseGPR(x_train[:m_sparse], noise)

    tick = now()
    model.train(x_train, y_train)
    tock = now()

    results = {
        'train_duration': tock - tick,
        'testing': {
            'test_set': test_model(model, x_test, y_test, cn_test),
            'train_set': test_model(model, x_train, y_train, cn_train),
        }
    }
    return model, results


def run():

    defaults = dict(
        fold=0, folds=5,                           # cross-val
        m_sparse=100, n_train=100, noise=0.01,     # model hypers
        n_max=12, l_max=6,                         # soap-hypers
        train_on_equal_cns=False,
    )

    passed_kwargs = parse_passed_kwargs(
        fold=int, m_sparse=int, n_train=int, n_max=int, l_max=int, noise=float, train_on_equal_cns=bool)
    config = {**defaults, **passed_kwargs}
    print(config)
    main(**config)
