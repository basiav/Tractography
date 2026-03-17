from sklearn.preprocessing import QuantileTransformer
import numpy as np

class Normalizations():
    def __init__(self, training_maps) -> None:
        """
        training_maps : array-like, shape (Nsubjects, Nvoxels)
            Training dataset (flattened tract density maps).
        template : nibabel.Nifti1Image
            Reference image providing the affine and header information.
        n_components : int or None
            Number of PCA components to retain.
        """

        self.training_maps = np.array(training_maps)
        
        # Preprocessing stats
        self.mu_train = None
        self.sigma_train = None
        self.Zmaps = None

        self.min_train = None
        self.max_train = None
        self.MinMaxmaps = None

        self.qt = None
        self.Qmaps = None
        self.qnorm_output_distribution = 'normal'

    def _train_Zscore(self):
        self.mu_train = self.training_maps.mean(axis=0)
        self.sigma_train = self.training_maps.std(axis=0)
        # Handle voxels outside the brain (all zeros) by avoiding division by zero
        self.sigma_train[self.sigma_train == 0] = 1.0
        self.Zmaps = (self.training_maps - self.mu_train) / self.sigma_train

    def _train_MinMax(self):
        """
        Scale training maps to [0, 1] using per-voxel min and max.
        """
        self.min_train = self.training_maps.min(axis=0)
        self.max_train = self.training_maps.max(axis=0)

        # Avoid division by zero when max == min
        denom = self.max_train - self.min_train
        denom[denom == 0] = 1.0

        self.MinMaxmaps = (self.training_maps - self.min_train) / denom


    def _train_QNorm(self):
        """
        Fit scikit-learn's QuantileTransformer to perform quantile-based
        normalization across subjects.

        Important notes
        ---------------
        - The typical quantile-normalization used in e.g. genomics makes the
        per-sample distributions identical by assigning each sample the
        average sorted values across samples. scikit-learn's
        QuantileTransformer maps features to follow a target distribution
        (uniform or normal). To apply an sklearn-style quantile transform
        *across subjects* (so that each subject's voxel intensity
        distribution is matched), we transpose the data, fit the
        QuantileTransformer treating each subject as a feature column,
        then transpose the result back.
        """
        output_distribution='normal'
        n_quantiles=1000
        random_state=0

        # Transpose so rows=voxels, cols=subjects => we transform per-subject distribution
        X = self.training_maps.T # shape: (n_voxels, n_subjects)

        # Fit QuantileTransformer on transposed data
        qt = QuantileTransformer(n_quantiles=min(n_quantiles, X.shape[0]),
        output_distribution=output_distribution,
        random_state=random_state,
        copy=True)
        X_q = qt.fit_transform(X)

        # Transpose back to (n_subjects, n_voxels)
        self.Qmaps = X_q.T
        self.qt = qt
        self.qnorm_output_distribution = output_distribution

    def Zscore(self, maps):
        """
        Apply Z-score normalization to new maps using training statistics.

        Parameters
        ----------
        maps : ndarray of shape (n_samples, n_voxels)
            New subject maps (already flattened) to normalize.

        Returns
        -------
        Zmaps : ndarray of shape (n_samples, n_voxels)
            Z-score normalized maps.

        Raises
        ------
        RuntimeError
            If the training statistics (mean and standard deviation)
            have not been computed. Call `fit_pca(norm='z_norm')` or `fit_transform_pca(norm='z_norm')` first.
        """
        if not hasattr(self, "mu_train") or not hasattr(self, "sigma_train"):
            raise RuntimeError("Must call fit_pca_maps() with 'z_norm' before applying Zscore.")
        return (maps - self.mu_train) / self.sigma_train
    
    def MinMax(self, maps):
        """
        Apply Min-Max normalization to new maps using training statistics.

        Parameters
        ----------
        maps : ndarray of shape (n_samples, n_voxels)
            New subject maps (already flattened) to normalize.

        Returns
        -------
        MinMaxmaps : ndarray of shape (n_samples, n_voxels)
            Min-Max normalized maps with values scaled to [0, 1].

        Raises
        ------
        RuntimeError
            If the training statistics (min and max) have not been computed.
            Call `fit_pca(norm='min_max')` or `fit_transform_pca(norm='min_max')` first.
        """ 
        if not hasattr(self, "min_train") or not hasattr(self, "max_train"):
            raise RuntimeError("Must call fit_pca_maps() with 'min_max' before applying MinMax normalization.")
        
        denom = self.max_train - self.min_train
        denom[denom == 0] = 1.0  # avoid division by zero
        return (maps - self.min_train) / denom


    def QNorm(self, maps):
        """
        Apply quantile normalization to new maps using the fitted
        scikit-learn QuantileTransformer.

        This method expects that `__train_QNorm()` has been called during
        training (e.g. via `fit_pca(norm='q_norm')` or by calling
        `__train_QNorm()` directly). The implementation transposes the
        input, applies the fitted QuantileTransformer so that each subject's
        voxel distribution is transformed to the target distribution, and
        then transposes the result back.

        Parameters
        ----------
        maps : ndarray, shape (n_samples, n_voxels)
        New subject maps (flattened) to normalize.

        Returns
        -------
        Qmaps : ndarray
        Quantile-normalized maps (same shape as input).

        Raises
        ------
        RuntimeError
        If the QuantileTransformer has not been fitted on training data.
        """
        if self.qt is None:
            raise RuntimeError("Must call fit_pca_maps(norm='q_norm') before applying QNorm.")

        X = np.array(maps)
        # Ensure 2D
        if X.ndim == 1:
            X = X[np.newaxis, :]

        # Transpose so rows=voxels, cols=subjects
        X_t = X.T
        X_q = self.qt.transform(X_t)
        return X_q.T