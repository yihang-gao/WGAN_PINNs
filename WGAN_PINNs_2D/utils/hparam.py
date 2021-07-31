"""
Hyper-parameter implementation
"""


class HParam(object):
    def __init__(self, **kwargs):
        self._parameters = {}

        for k, v in kwargs.items():
            self._parameters[k] = v

    def set_param(self, name, value):
        self._parameters[name] = value

    def add_params(self, **kwargs):
        for k, v in kwargs.items():
            self._parameters[k] = v

    @property
    def params(self):
        return self._parameters

    def __getitem__(self, key):
        return self._parameters[key]

    def __str__(self) -> str:
        parameters = [str(k) + ': ' + str(v) for k, v in self._parameters.items()]
        hp_str = "\r\n".join(parameters)

        return hp_str
