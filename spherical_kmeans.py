import warnings
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.utils.sparsefuncs_fast import assign_rows_csr
from sklearn.cluster._k_means_fast import _mini_batch_update_csr
from sklearn.utils.validation import _check_sample_weight
from sklearn.utils import check_array, check_random_state
from sklearn.utils.extmath import row_norms
import scipy.sparse as sp
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster.k_means_ import (
    _init_centroids,
    _labels_inertia,
    _tolerance,
    _mini_batch_step,
    _mini_batch_convergence
)


def _check_normalize_sample_weight(sample_weight, X):
    """Set sample_weight if None, and check for correct dtype"""

    sample_weight_was_none = sample_weight is None

    sample_weight = _check_sample_weight(sample_weight, X)

    if not sample_weight_was_none:
        # normalize the weights to sum up to n_samples
        # an array of 1 (i.e. samples_weight is None) is already normalized
        n_samples = len(sample_weight)
        scale = n_samples / sample_weight.sum()
        sample_weight *= scale
    return sample_weight




def _mini_batch_spherical_step(X, sample_weight, x_squared_norms, centers, weight_sums,
                               old_center_buffer, compute_squared_diff,
                               distances, random_reassign=False,
                               random_state=None, reassignment_ratio=.01,
                               verbose=False):
    """Incremental update of the centers for the Minibatch K-Means algorithm.
    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        The original data array.
    sample_weight : array-like, shape (n_samples,)
        The weights for each observation in X.
    x_squared_norms : array, shape (n_samples,)
        Squared euclidean norm of each data point.
    centers : array, shape (k, n_features)
        The cluster centers. This array is MODIFIED IN PLACE
    counts : array, shape (k,)
         The vector in which we keep track of the numbers of elements in a
         cluster. This array is MODIFIED IN PLACE
    distances : array, dtype float, shape (n_samples), optional
        If not None, should be a pre-allocated array that will be used to store
        the distances of each sample to its closest center.
        May not be None when random_reassign is True.
    random_state : int, RandomState instance or None (default)
        Determines random number generation for centroid initialization and to
        pick new clusters amongst observations with uniform probability. Use
        an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.
    random_reassign : boolean, optional
        If True, centers with very low counts are randomly reassigned
        to observations.
    reassignment_ratio : float, optional
        Control the fraction of the maximum number of counts for a
        center to be reassigned. A higher value means that low count
        centers are more likely to be reassigned, which means that the
        model will take longer to converge, but should converge in a
        better clustering.
    verbose : bool, optional, default False
        Controls the verbosity.
    compute_squared_diff : bool
        If set to False, the squared diff computation is skipped.
    old_center_buffer : int
        Copy of old centers for monitoring convergence.
    Returns
    -------
    inertia : float
        Sum of squared distances of samples to their closest cluster center.
    squared_diff : numpy array, shape (n_clusters,)
        Squared distances between previous and updated cluster centers.
    """
    # Perform label assignment to nearest centers
    nearest_center, inertia = _labels_inertia(X, sample_weight,
                                              x_squared_norms, centers,
                                              distances=distances)

    if random_reassign and reassignment_ratio > 0:
        random_state = check_random_state(random_state)
        # Reassign clusters that have very low weight
        to_reassign = weight_sums < reassignment_ratio * weight_sums.max()
        # pick at most .5 * batch_size samples as new centers
        if to_reassign.sum() > .5 * X.shape[0]:
            indices_dont_reassign = \
                np.argsort(weight_sums)[int(.5 * X.shape[0]):]
            to_reassign[indices_dont_reassign] = False
        n_reassigns = to_reassign.sum()
        if n_reassigns:
            # Pick new clusters amongst observations with uniform probability
            new_centers = random_state.choice(X.shape[0], replace=False,
                                              size=n_reassigns)
            if verbose:
                print("[MiniBatchKMeans] Reassigning %i cluster centers."
                      % n_reassigns)

            if sp.issparse(X) and not sp.issparse(centers):
                assign_rows_csr(
                    X, new_centers.astype(np.intp, copy=False),
                    np.where(to_reassign)[0].astype(np.intp, copy=False),
                    centers)
            else:
                centers[to_reassign] = X[new_centers]
        # reset counts of reassigned centers, but don't reset them too small
        # to avoid instant reassignment. This is a pretty dirty hack as it
        # also modifies the learning rates.
        weight_sums[to_reassign] = np.min(weight_sums[~to_reassign])

    # implementation for the sparse CSR representation completely written in
    # cython
    if sp.issparse(X):
        return inertia, _mini_batch_update_csr(
            X, sample_weight, x_squared_norms, centers, weight_sums,
            nearest_center, old_center_buffer, compute_squared_diff)

    # dense variant in mostly numpy (not as memory efficient though)
    k = centers.shape[0]
    squared_diff = 0.0
    for center_idx in range(k):
        # find points from minibatch that are assigned to this center
        center_mask = nearest_center == center_idx
        wsum = sample_weight[center_mask].sum()

        if wsum > 0:
            if compute_squared_diff:
                old_center_buffer[:] = centers[center_idx]

            # inplace remove previous count scaling
            centers[center_idx] *= weight_sums[center_idx]

            # inplace sum with new points members of this cluster
            centers[center_idx] += \
                np.sum(X[center_mask] *
                       sample_weight[center_mask, np.newaxis], axis=0)

            # unit-normalize for spherical k-means
            centers[center_idx] = normalize(centers[center_idx, None])[:, 0]

            # update the squared diff if necessary
            if compute_squared_diff:
                diff = centers[center_idx].ravel() - old_center_buffer.ravel()
                squared_diff += np.dot(diff, diff)

    return inertia, squared_diff


class MiniBatchSphericalKMeans(MiniBatchKMeans):

    def fit(self, X, y=None, sample_weight=None):
        """Compute the centroids on X by chunking it into mini-batches.
        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            Training instances to cluster. It must be noted that the data
            will be converted to C ordering, which will cause a memory copy
            if the given data is not C-contiguous.
        y : Ignored
            Not used, present here for API consistency by convention.
        sample_weight : array-like, shape (n_samples,), optional
            The weights for each observation in X. If None, all observations
            are assigned equal weight (default: None).
        Returns
        -------
        self
        """
        random_state = check_random_state(self.random_state)
        # unit-normalize for spherical k-means
        X = normalize(X)
        X = check_array(X, accept_sparse="csr", order='C',
                        dtype=[np.float64, np.float32])
        n_samples, n_features = X.shape
        if n_samples < self.n_clusters:
            raise ValueError("n_samples=%d should be >= n_clusters=%d"
                             % (n_samples, self.n_clusters))

        sample_weight = _check_normalize_sample_weight(sample_weight, X)

        n_init = self.n_init
        if hasattr(self.init, '__array__'):
            self.init = np.ascontiguousarray(self.init, dtype=X.dtype)
            if n_init != 1:
                warnings.warn(
                    'Explicit initial center position passed: '
                    'performing only one init in MiniBatchKMeans instead of '
                    'n_init=%d'
                    % self.n_init, RuntimeWarning, stacklevel=2)
                n_init = 1

        x_squared_norms = row_norms(X, squared=True)

        if self.tol > 0.0:
            tol = _tolerance(X, self.tol)

            # using tol-based early stopping needs the allocation of a
            # dedicated before which can be expensive for high dim data:
            # hence we allocate it outside of the main loop
            old_center_buffer = np.zeros(n_features, dtype=X.dtype)
        else:
            tol = 0.0
            # no need for the center buffer if tol-based early stopping is
            # disabled
            old_center_buffer = np.zeros(0, dtype=X.dtype)

        distances = np.zeros(self.batch_size, dtype=X.dtype)
        n_batches = int(np.ceil(float(n_samples) / self.batch_size))
        n_iter = int(self.max_iter * n_batches)

        init_size = self.init_size
        if init_size is None:
            init_size = 3 * self.batch_size
        if init_size > n_samples:
            init_size = n_samples
        self.init_size_ = init_size

        validation_indices = random_state.randint(0, n_samples, init_size)
        X_valid = X[validation_indices]
        sample_weight_valid = sample_weight[validation_indices]
        x_squared_norms_valid = x_squared_norms[validation_indices]

        # perform several inits with random sub-sets
        best_inertia = None
        for init_idx in range(n_init):
            if self.verbose:
                print("Init %d/%d with method: %s"
                      % (init_idx + 1, n_init, self.init))
            weight_sums = np.zeros(self.n_clusters, dtype=sample_weight.dtype)

            # TODO: once the `k_means` function works with sparse input we
            # should refactor the following init to use it instead.

            # Initialize the centers using only a fraction of the data as we
            # expect n_samples to be very large when using MiniBatchKMeans
            cluster_centers = _init_centroids(
                X, self.n_clusters, self.init,
                random_state=random_state,
                x_squared_norms=x_squared_norms,
                init_size=init_size)

            cluster_centers = normalize(cluster_centers)

            # Compute the label assignment on the init dataset
            _mini_batch_step(
                X_valid, sample_weight_valid,
                x_squared_norms[validation_indices], cluster_centers,
                weight_sums, old_center_buffer, False, distances=None,
                verbose=self.verbose)

            cluster_centers = normalize(cluster_centers)

            # Keep only the best cluster centers across independent inits on
            # the common validation set
            _, inertia = _labels_inertia(X_valid, sample_weight_valid,
                                         x_squared_norms_valid,
                                         cluster_centers)
            if self.verbose:
                print("Inertia for init %d/%d: %f"
                      % (init_idx + 1, n_init, inertia))
            if best_inertia is None or inertia < best_inertia:
                self.cluster_centers_ = cluster_centers
                self.counts_ = weight_sums
                best_inertia = inertia

        # Empty context to be used inplace by the convergence check routine
        convergence_context = {}

        # Perform the iterative optimization until the final convergence
        # criterion
        for iteration_idx in range(n_iter):
            # Sample a minibatch from the full dataset
            minibatch_indices = random_state.randint(
                0, n_samples, self.batch_size)

            # Perform the actual update step on the minibatch data
            self.cluster_centers_ = normalize(self.cluster_centers_)
            batch_inertia, centers_squared_diff = _mini_batch_step(
                X[minibatch_indices], sample_weight[minibatch_indices],
                x_squared_norms[minibatch_indices],
                self.cluster_centers_, self.counts_,
                old_center_buffer, tol > 0.0, distances=distances,
                # Here we randomly choose whether to perform
                # random reassignment: the choice is done as a function
                # of the iteration index, and the minimum number of
                # counts, in order to force this reassignment to happen
                # every once in a while
                random_reassign=((iteration_idx + 1)
                                 % (10 + int(self.counts_.min())) == 0),
                random_state=random_state,
                reassignment_ratio=self.reassignment_ratio,
                verbose=self.verbose)
            self.cluster_centers_ = normalize(self.cluster_centers_)

            # Monitor convergence and do early stopping if necessary
            if _mini_batch_convergence(
                    self, iteration_idx, n_iter, tol, n_samples,
                    centers_squared_diff, batch_inertia, convergence_context,
                    verbose=self.verbose):
                break

        self.n_iter_ = iteration_idx + 1

        if self.compute_labels:
            self.labels_, self.inertia_ = \
                self._labels_inertia_minibatch(X, sample_weight)

        return self

    def partial_fit(self, X, y=None, sample_weight=None):
        """Update k means estimate on a single mini-batch X.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Coordinates of the data points to cluster. It must be noted that
            X will be copied if it is not C-contiguous.
        y : Ignored
            Not used, present here for API consistency by convention.
        sample_weight : array-like, shape (n_samples,), optional
            The weights for each observation in X. If None, all observations
            are assigned equal weight (default: None).
        Returns
        -------
        self
        """

        X = check_array(X, accept_sparse="csr", order="C",
                        dtype=[np.float64, np.float32])
        n_samples, n_features = X.shape
        if hasattr(self.init, '__array__'):
            self.init = np.ascontiguousarray(self.init, dtype=X.dtype)

        if n_samples == 0:
            return self

        # unit-normalize for spherical k-means
        X = normalize(X)

        sample_weight = _check_normalize_sample_weight(sample_weight, X)

        x_squared_norms = row_norms(X, squared=True)
        self.random_state_ = getattr(self, "random_state_",
                                     check_random_state(self.random_state))
        if (not hasattr(self, 'counts_')
                or not hasattr(self, 'cluster_centers_')):
            # this is the first call partial_fit on this object:
            # initialize the cluster centers
            self.cluster_centers_ = _init_centroids(
                X, self.n_clusters, self.init,
                random_state=self.random_state_,
                x_squared_norms=x_squared_norms, init_size=self.init_size)

            self.counts_ = np.zeros(self.n_clusters,
                                    dtype=sample_weight.dtype)
            random_reassign = False
            distances = None
        else:
            # The lower the minimum count is, the more we do random
            # reassignment, however, we don't want to do random
            # reassignment too often, to allow for building up counts
            random_reassign = self.random_state_.randint(
                10 * (1 + self.counts_.min())) == 0
            distances = np.zeros(X.shape[0], dtype=X.dtype)

        self.cluster_centers_ = normalize(self.cluster_centers_)

        _mini_batch_spherical_step(X, sample_weight, x_squared_norms,
                                   self.cluster_centers_, self.counts_,
                                   np.zeros(0, dtype=X.dtype), 0,
                                   random_reassign=random_reassign, distances=distances,
                                   random_state=self.random_state_,
                                   reassignment_ratio=self.reassignment_ratio,
                                   verbose=self.verbose)
        self.cluster_centers_ = normalize(self.cluster_centers_)

        if self.compute_labels:
            self.labels_, self.inertia_ = _labels_inertia(
                X, sample_weight, x_squared_norms, self.cluster_centers_)

        return self