from abc import ABC, abstractmethod


class Transformer(ABC):
    @abstractmethod
    def transform_instance(self, instance):
        raise NotImplementedError

