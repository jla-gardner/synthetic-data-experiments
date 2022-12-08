from experiments import parse_passed_kwargs
from experiments.hypers import sample
from dkl import main, device

hyper_spec = dict(
    lr={'method': 'log', 'min': 0.0001, 'max': 0.001, 'type': float},
    depth={'method': 'uniform', 'min': 2, 'max': 6, 'type': int},
    width={'method': 'log', 'min': 50, 'max': 10_000, 'type': int},
    last_width={'method': 'log', 'min': 5, 'max': 500, 'type': int},
    activation={'method': 'choice', 'options': ['celu', 'tanh', 'relu']},
    epochs={'method': 'uniform', 'min': 200, 'max': 1000, 'type': int},
    activate_last={'method': 'choice', 'options': [True, False]},
    m_sparse={'method': 'log', 'min': 50, 'max': 4000, 'type': int},
    batchsize={'method': 'log', 'min': 500, 'max': 8000, 'type': int},
    early_stopping={'method': 'log', 'min': 20, 'max': 400, 'type': int},
)

def run():
    config = dict(
        folds=5,
        n_train=1_000,
        device_type=device.type, decay=0.9985,
        n_max=11, l_max=4, train_on_equal_cns=False,
        **sample(hyper_spec),
    )

    passed = parse_passed_kwargs(
        epochs=int, depth=int, width=int, decay=float, lr=float, n_train=int,
        n_max=int, l_max=int, early_stopping=int,
    )

    config = {**config, **passed}
    print(passed)
    print(config)
    
    # for fold in range(5):
    main(fold=0, **config)

