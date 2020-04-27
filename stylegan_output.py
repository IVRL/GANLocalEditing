import torch
from ptutils import MultiResolutionStore


def add_batch_dim(t):
    if torch.is_tensor(t) and t.dim() == 1:
        return t.unsqueeze(0)
    return t


class GANOutputs:

    @staticmethod
    def from_seed(seed_indices, seed, dlatent=512):

        if type(seed_indices) is int:
            seed_indices = tuple(range(seed_indices))
        else:
            seed_indices = seed_indices

        if seed is not None:
            torch.manual_seed(seed)


        assert seed_indices is not None

        z = torch.randn(max(seed_indices) + 1, dlatent)[seed_indices, :]

        gs = GANOutputs()
        gs.seed = seed
        gs.seed_indices = seed_indices
        gs.z = z

        return gs


    def __len__(self):
        if hasattr(self, 'z'):
            return len(self.z)
        if hasattr(self, 'ys'):
            return len(self.ys[0])
        return None


    def __getitem__(self, item):
        new = GANOutputs()
        for k, v in self.__dict__.items():
            if torch.is_tensor(v):
                new.__dict__[k] = add_batch_dim(v[item])

            elif type(v) is list and torch.is_tensor(v[0]):
                new.__dict__[k] = [add_batch_dim(vv[item]) for vv in v]

            elif type(v) is dict:
                new.__dict__[k] = {kk: add_batch_dim(vv[item]) for kk, vv in v.items()}

            elif type(v) is MultiResolutionStore:
                new.__dict__[k] = MultiResolutionStore(v.get()[item])

        return new


    def __repr__(self):
        ret =  '{} GANOutput(s)'.format(len(self))# + str((self.seed, self.seed_indices))
        if hasattr(self, 'seed'):
            ret += ': (seed: {}, indices: {})'.format(self.seed, self.seed_indices)
        return ret
