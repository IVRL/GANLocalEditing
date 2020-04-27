'''
To download pickled instances for FFHQ and LSUN-Bedrooms, visit: https://drive.google.com/open?id=1GYzEzOCaI8FUS6JHdt6g9UfNTmpO08Tt
'''

import torch
import ptutils
from spherical_kmeans import MiniBatchSphericalKMeans


def one_hot(a, n):
    import numpy as np
    b = np.zeros((a.size, n))
    b[np.arange(a.size), a] = 1
    return b

class FactorCatalog:

    def __init__(self, k, random_state=0, factorization=None, **kwargs):

        if factorization is None:
            factorization = MiniBatchSphericalKMeans

        self._factorization = factorization(n_clusters=k, random_state=random_state, **kwargs)

        self.annotations = {}

    def _preprocess(self, X):
        X_flat = ptutils.partial_flat(X)
        return X_flat

    def _postprocess(self, labels, X, raw):
        heatmaps = torch.from_numpy(one_hot(labels, self._factorization.cluster_centers_.shape[0])).float()
        heatmaps = ptutils.partial_unflat(heatmaps, N=X.shape[0], H=X.shape[-1])
        if raw:
            heatmaps = ptutils.MultiResolutionStore(heatmaps, 'nearest')
            return heatmaps
        else:
            heatmaps = ptutils.MultiResolutionStore(torch.cat([(heatmaps[:, v].sum(1, keepdim=True)) for v in
                        self.annotations.values()], 1), 'nearest')
            labels = list(self.annotations.keys())

            return heatmaps, labels

    def fit_predict(self, X, raw=False):
        self._factorization.fit(self._preprocess(X))
        labels = self._factorization.labels_
        return self._postprocess(labels, X, raw)

    def predict(self, X, raw=False):
        labels = self._factorization.predict(self._preprocess(X))
        return self._postprocess(labels, X, raw)

    def __repr__(self):
        header = '{} catalog:'.format(type(self._factorization))
        return '{}\n\t{}'.format(header, self.annotations)

