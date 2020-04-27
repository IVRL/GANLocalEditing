import torch
from collections import OrderedDict

def ordereddict_insert(d, item, index=None, key=None):
    assert (index is not None) != (key is not None)
    k = list(d.keys())
    v = list(d.values())

    if key is not None:
        index = k.index(key)+1

    k.insert(index, item[0])
    v.insert(index, item[1])

    return OrderedDict(zip(k,v))

def add_after(model, key, new_key, new_module):
        assert key in model._modules
        model._modules = ordereddict_insert(model._modules, (new_key, new_module), key=key)


def partial_flat(x):
    flat =  x.permute(0,2,3,1).contiguous().view(-1,x.shape[1])
    flat.original_shape = x.shape
    return flat

def partial_unflat(x, N=None, H=None, W=None):
    assert x.dim() == 2
    C = x.shape[1]
    if N is None:
        N, C, H, W = x.original_shape
    if W is None:
        W = H
    assert N is not None and H is not None and W is not None
    return x.view(N, H, W, C).permute(0,3,1,2)



class MultiResolutionStore:


    def __init__(self, item=None, interpolation_mode='bilinear'):
        self._data = {}
        self._res = None
        if item is not None:
            self._res = item.shape[-1]
            self._data[self._res] = item

        self._cuda = False
        self.interpolation_mode = interpolation_mode

    def cuda(self):
        self._cuda = True
        return self

    def cpu(self):
        self._cuda = False
        return self

    def get(self, res=None, make=True):
        if res == None:
            res = self._res

        if res not in self and make:
            self.make(res)

        ret = self._data[res]
        if self._cuda:
            ret = ret.cuda()

        return ret

    def __getitem__(self, res):
        return self.get(res, make=False)

    def __contains__(self, res):
        return res in self._data

    def __len__(self):
        return len(self._data)

    def resolutions(self):
        return (res for res in self._data.keys())

    def __repr__(self):
        return 'MultiResolutionStore {}: {}'.format(self._data[self._res].shape, list(self.resolutions()))

    def make(self, res):
        self._data[res] = self._resize(res)

    def _resize(self, res):
        assert type(res) is int
        return torch.nn.functional.interpolate(self._data[self._res], size=(res, res), mode=self.interpolation_mode)






