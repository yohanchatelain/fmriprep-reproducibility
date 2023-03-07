import pickle


class Statistic:

    def __init__(self, name='output.pkl'):
        self._data = list()
        self._name = name

    def set_name(self, name):
        if not name.endswith('.pkl'):
            self._name = name + '.pkl'
        else:
            self._name = name

    def append(self, **kwargs):
        self._data.append(kwargs)

    def dump(self):
        with open(self._name, 'wb') as fo:
            pickle.dump(self._data, fo)


stats_collect = Statistic()
