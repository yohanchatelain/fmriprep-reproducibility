from abc import ABC, abstractmethod


class Sample(ABC):
    @abstractmethod
    def get_subsample(self, indexes):
        pass

    @abstractmethod
    def get_size(self):
        pass

    