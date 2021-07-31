"""
Registry mechanism
"""


class Registry(object):

    def __init__(self, registry_name):
        self._registry_name = registry_name
        self._registry = {}

    def register(self, name=None):
        def decorator(class_or_func):
            if name is None:
                self._registry[class_or_func.__name__] = class_or_func
            else:
                self._registry[name] = class_or_func

            return class_or_func

        return decorator

    def get(self, name):
        return self._registry[name]

    def keys(self):
        return self._registry.keys()

    def __contains__(self, key):
        return self._registry.__contains__(key)

    def __iter__(self):
        return self._registry.__iter__()

    def __str__(self) -> str:
        return self._registry.__str__()


class Registries(object):

    def __init__(self):
        raise RuntimeError('Registries is not allowed to be instantiated')

    datasets = Registry('datasets')
    problems = Registry('problems')
    hparams = Registry('hparams')


register_dataset = Registries.datasets.register
register_problem = Registries.problems.register
register_hparam = Registries.hparams.register

get_dataset = Registries.datasets.get
get_problem = Registries.problems.get
get_hparam = Registries.hparams.get

datasets = Registries.datasets
problems = Registries.problems
hparams = Registries.hparams
