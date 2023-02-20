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

    def append(self, dataset, subject, confidence, sample_size,
               target, fvr, method, fwh, nb_round=None, kth_round=None):
        _row = dict(dataset=dataset,
                    subject=subject,
                    confidence=confidence,
                    sample_size=sample_size,
                    target=target,
                    method=method,
                    fwh=fwh,
                    fvr=fvr,
                    nb_round=nb_round,
                    kth_round=kth_round)
        self._data.append(_row)

    def dump(self):
        with open(self._name, 'wb') as fo:
            pickle.dump(self._data, fo)


stats_collect = Statistic()
