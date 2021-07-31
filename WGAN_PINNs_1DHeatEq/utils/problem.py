"""
Problem template
"""

import abc


class Problem(object):
    def __init__(self, hparam):
        self.hparam = hparam

    @abc.abstractmethod
    def load_data(self):
        pass

    @abc.abstractmethod
    def train_model(self):
        pass

    @abc.abstractmethod
    def test_model(self):
        pass


