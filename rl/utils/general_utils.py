import collections
import itertools
import random
import time
from functools import reduce

import numpy as np
import torch


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


class Until:
    def __init__(self, until, action_repeat=1):
        self._until = until
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._until is None:
            return True
        until = self._until // self._action_repeat
        return step < until


class Every:
    def __init__(self, every, action_repeat=1):
        self._every = every
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._every is None:
            return False
        every = self._every // self._action_repeat
        if step % every == 0:
            return True
        return False


class Timer:
    def __init__(self):
        self._start_time = time.time()
        self._last_time = time.time()

    def reset(self):
        elapsed_time = time.time() - self._last_time
        self._last_time = time.time()
        total_time = time.time() - self._start_time
        return elapsed_time, total_time

    def total_time(self):
        return time.time() - self._start_time
    

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, digits=None):
        """
        :param digits: number of digits returned for average value
        """
        self._digits = digits
        self.reset()

    def reset(self):
        self.val = 0
        self._avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self._avg = self.sum / max(1, self.count)

    @property
    def avg(self):
        if self._digits is not None:
            return np.round(self._avg, self._digits)
        else:
            return self._avg


class RecursiveAverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = 0

    def update(self, val):
        self.val = val
        if self.sum is None:
            self.sum = val
        else:
            self.sum = map_recursive_list(lambda x, y: x + y, [self.sum, val])
        self.count += 1
        self.avg = map_recursive(lambda x: x / self.count, self.sum)



class AttrDict(dict):
    __setattr__ = dict.__setitem__

    def __getattr__(self, attr):
        # Take care that getattr() raises AttributeError, not KeyError.
        # Required e.g. for hasattr(), deepcopy and OrderedDict.
        try:
            return self.__getitem__(attr)
        except KeyError:
            raise AttributeError("Attribute %r not found" % attr)

    def __getstate__(self):
        return self

    def __setstate__(self, d):
        self = d

def map_dict(fn, d):
    """takes a dictionary and applies the function to every element"""
    return type(d)(map(lambda kv: (kv[0], fn(kv[1])), d.items()))

def map_recursive(fn, tensors):
    return make_recursive(fn)(tensors)

def map_recursive_list(fn, tensors):
    return make_recursive_list(fn)(tensors)

def make_recursive(fn, *argv, **kwargs):
    """ Takes a fn and returns a function that can apply fn on tensor structure
     which can be a single tensor, tuple or a list. """
    
    def recursive_map(tensors):
        if tensors is None:
            return tensors
        elif isinstance(tensors, list) or isinstance(tensors, tuple):
            return type(tensors)(map(recursive_map, tensors))
        elif isinstance(tensors, dict):
            return type(tensors)(map_dict(recursive_map, tensors))
        elif isinstance(tensors, torch.Tensor) or isinstance(tensors, np.ndarray):
            return fn(tensors, *argv, **kwargs)
        else:
            try:
                return fn(tensors, *argv, **kwargs)
            except Exception as e:
                print("The following error was raised when recursively applying a function:")
                print(e)
                raise ValueError("Type {} not supported for recursive map".format(type(tensors)))

    return recursive_map

def make_recursive_list(fn):
    """ Takes a fn and returns a function that can apply fn across tuples of tensor structures,
     each of which can be a single tensor, tuple or a list. """

    def recursive_map(tensors):
        if tensors is None:
            return tensors
        elif isinstance(tensors[0], list) or isinstance(tensors[0], tuple):
            return type(tensors[0])(map(recursive_map, zip(*tensors)))
        elif isinstance(tensors[0], dict):
            return map_dict(recursive_map, listdict2dictlist(tensors))
        elif isinstance(tensors[0], torch.Tensor):
            return fn(*tensors)
        else:
            try:
                return fn(*tensors)
            except Exception as e:
                print("The following error was raised when recursively applying a function:")
                print(e)
                raise ValueError("Type {} not supported for recursive map".format(type(tensors)))

    return recursive_map

def listdict2dictlist(LD):
    """ Converts a list of dicts to a dict of lists """
    
    # Take intersection of keys
    keys = reduce(lambda x,y: x & y, (map(lambda d: d.keys(), LD)))
    return AttrDict({k: [dic[k] for dic in LD] for k in keys})

def joinListDictList(LDL):
    """Joins a list of dictionaries that contain lists."""
    DLL = listdict2dictlist(LDL)
    return type(LDL[0])({k: np.array(list(itertools.chain.from_iterable(DLL[k]))) for k in DLL})

def joinListDict(LD):
    """Joins a list of dictionaries that contain lists."""
    DL = listdict2dictlist(LD)
    return type(LD[0])({k: np.array(DL[k]) for k in DL})

def joinListList(LL):
    """Joins a list of dictionaries that contain lists."""
    return type(LL[0])(itertools.chain.from_iterable(LL))

def obj2np(obj):
    """Wraps an object into an np.array."""
    ar = np.zeros((1,), dtype=np.object_)
    ar[0] = obj
    return ar

def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def prefix_dict(d, prefix):
    """Adds the prefix to all keys of dict d."""
    return type(d)({prefix+k: v for k, v in d.items()})

def np2obj(np_array):
    if isinstance(np_array, list) or np_array.size > 1:
        return [e[0] for e in np_array]
    else:
        return np_array[0]
    
def str2int(str):
    try:
        return int(str)
    except ValueError:
        return None
    
def get_last_argmax(array):
    b = array[::-1]
    idx = len(b) - np.argmax(array) - 1
    return idx
