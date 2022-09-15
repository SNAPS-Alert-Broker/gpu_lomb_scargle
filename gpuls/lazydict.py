"""From: https://stackoverflow.com/a/47212782
"""

from collections.abc import Mapping

class LazyDict(Mapping):
    def __init__(self, *args, **kw):
        self._raw_dict = dict(*args, **kw)

    def __getitem__(self, key):
        value = self._raw_dict.__getitem__(key)
        if isinstance(value, tuple):
            function, args = value
            value = function(*args)
            self._raw_dict.__setitem__(key, value)
        return value

    def __iter__(self):
        return iter(self._raw_dict)

    def __len__(self):
        return len(self._raw_dict)
