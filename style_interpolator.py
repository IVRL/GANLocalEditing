import torch
from copy import deepcopy

def _repeat_batch(t, n):
    rep = torch.ones(t.dim())
    rep[0] = n
    return t.repeat(tuple(rep.int()))

class StyleInterpolator:

    def __init__(self, catalog, bias=False):
        self._catalog = catalog
        self._catalog_labels = list(catalog.annotations.keys())
        self._bias = bias

        self._cumsum = []
        self._order = []
        for layer in range(len(self._catalog.M)):
            c = torch.zeros_like(self._catalog.M[layer])
            o = torch.zeros_like(self._catalog.M[layer]).long()
            for label_i in range(self._catalog.M[layer].shape[0]):
                in_mse = self._catalog.M[layer][label_i]
                order = torch.argsort(in_mse, descending=True)
                o[label_i] = order
                c[label_i] = torch.cumsum(1 - in_mse[order], dim=0)
            self._order.append(o)
            self._cumsum.append(c)

        self._cache = {}

    def _interpolate_y(self, y1, y2, q):
        if torch.is_tensor(q):


            q = q.to(y1.device)

        if len(y1) == 1 and len(y2) > 1:
            y1 = _repeat_batch(y1, len(y2))

        if y1.dim() == 5:  # batch X {scale, bias} X channels X 1 X 1

            if q.dim() == 1:  # channels
                q = q[:, None, None]

            y3 = deepcopy(y1)
            y3[:, 0] = (1 - q) * y1[:, 0] + q * y2[:, 0]

            if hasattr(self, '_bias') and self._bias:
                y3[:, 1] = (1 - q) * y1[:, 1] + q * y2[:, 1]

        elif y1.dim() == 2:  # batch X channels  (only scale)
            y3 = (1 - q) * y1 + q * y2

        return y3


    def _get_q(self, layer, label, rho, epsilon):
        if (layer, label, rho, epsilon) not in self._cache:
            label_i = self._catalog_labels.index(label)
            in_mse = self._catalog.M[layer][label_i]
            order = self._order[layer][label_i]
            cumsum = self._cumsum[layer][label_i]
            q = torch.zeros_like(in_mse)
            q[order[cumsum < epsilon]] = 1
            q *= (in_mse > rho).float()
            self._cache[layer, label, rho, epsilon] = q
        return self._cache[layer, label, rho, epsilon]



    def interpolate_ys(self, ys1, ys2, label, rho, epsilon):
        assert len(ys1) == len(ys2)

        if type(rho) is float or type(rho) is int:
            rho = [rho] * len(ys1)

        if type(epsilon) is float or type(epsilon) is int:
            epsilon = [epsilon] * len(ys1)

        return [self._interpolate_y(ys1[i], ys2[i], self._get_q(i, label, rho[i], epsilon[i]))
                for i in range(len(ys1))]