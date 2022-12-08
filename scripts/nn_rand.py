from experiments.hypers import sample
from experiments import parse_passed_kwargs
from nn import main, device

hyper_spec = dict(
    lr={'method': 'log', 'min': 0.0001, 'max': 0.001, 'type': float},
    depth={'method': 'uniform', 'min': 2, 'max': 7, 'type': int},
    width={'method': 'log', 'min': 200, 'max': 20_000, 'type': int},
    activation={'method': 'choice', 'options': ['celu', 'tanh', 'relu']},
    epochs={'method': 'uniform', 'min': 2_500, 'max': 5000, 'type': int},
    decay={'method': 'log', 'min': 0.985, 'max': 1},
    early_stopping={'method': 'log', 'min': 40, 'max': 400, 'type': int}
)


def run():
    config = dict(
        folds=5,
        n_train=1_000_000, batchsize=1000,
        device_type=device.type,
        train_on_equal_cns=False,
        **sample(hyper_spec),
    )

    passed = parse_passed_kwargs(
        epochs=int, depth=int, width=int, decay=float, lr=float, n_train=int,
        n_max=int, l_max=int, batchsize=int, early_stopping=int
    )

    config = {**config, **passed}
    print(config)

    main(fold=0, **config)
