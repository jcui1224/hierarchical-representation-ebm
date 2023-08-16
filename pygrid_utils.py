import itertools
import datetime

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def to_named_dict(ns):
    d = AttrDict()
    for (k, v) in zip(ns.__dict__.keys(), ns.__dict__.values()):
        d[k] = v
    return d

def merge_dicts(a, b, c):
    d = {}
    d.update(a)
    d.update(b)
    d.update(c)
    return d
