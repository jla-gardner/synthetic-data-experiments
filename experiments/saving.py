from datetime import datetime
import functools
import json
from pathlib import Path
from warnings import warn

from experiments.util import matches_template

_top_dir = Path(__file__).parent.parent
__experiments_dir = _top_dir / "results"
__models_dir = _top_dir / ".models"


def now(): return datetime.now().strftime("%Y/%m/%d %H:%M:%S")


def name(model_type, **info):
    return f"{model_type}-" + "-".join(str(info[k]) for k in sorted(info))


def model_path(model_type, config):
    _id = name(model_type, **config)
    return __models_dir / _id


def experiment(model_type):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(**config):
            _id = name(model_type, **config)

            model, results = func(**config)

            model.save(__models_dir / _id)
            add_experiment(
                _id + ".json", model_type=model_type, date=now(),
                config=config, results=results
            )

        return wrapper
    return decorator


def load_experiments():
    _exps = {}
    for f in __experiments_dir.iterdir():
        if not f.is_file():
            continue

        try:
            _exps[f.name] = json.loads(f.read_text())
        except:
            raise Warning(f"{f.name} not a valid json file")

    return _exps


__experiments = load_experiments()


def save_experiments():
    for name, expmt in __experiments.items():
        (__experiments_dir / name).write_text(json.dumps(expmt, indent=2))


def add_experiment(id=None, **kwargs):
    exmpt = kwargs
    if id in __experiments:
        warn("Overwriting existing experiment")
    __experiments[id] = exmpt

    save_experiments()

    return exmpt


def experiments_matching(**template):
    __experiments = load_experiments()
    return [*filter(matches_template(template), __experiments.values())]
