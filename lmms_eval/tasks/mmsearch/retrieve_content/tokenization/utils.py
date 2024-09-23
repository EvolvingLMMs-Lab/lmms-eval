import json
import pickle


class PickleWriteable:
    """Mixin for persisting an instance with pickle."""

    def save(self, path):
        try:
            with open(path, 'wb') as f:
                pickle.dump(self, f)
        except (pickle.PickleError, OSError) as e:
            raise IOError('Unable to save {} to path: {}'.format(self.__class__.__name__, path)) from e

    @classmethod
    def load(cls, path):
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except (pickle.PickleError, OSError) as e:
            raise IOError('Unable to load {} from path: {}'.format(cls.__name__, path)) from e
