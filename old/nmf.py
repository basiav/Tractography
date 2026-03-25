from sklearn.decomposition import NMF

from normalize import Normalizations
# from masking import Masking
from masking_new import MaskingNew

import numpy as np
import matplotlib.pyplot as plt
import os

from TractDensityComponents import TD_PCA


class NMFProjections(MaskingNew, NMF):
    def __init__(self, training_maps, template, binary_mask=None, n_components=None, max_iter: int = 10) -> None:
            """
            training_maps : array-like, shape (Nsubjects, Nvoxels)
                Training dataset (flattened tract density maps).
            template : nibabel.Nifti1Image
                Reference image providing the affine and header information.
            binary_mask : nibabel.Nifti1Image
                Binary mask with the voxels to use for the PCA fit.
            n_components : int or None
                Number of PCA components to retain.
            """

            self.shape = template.get_fdata().shape
            self.template = template

            # To mask data
            # td_pca = TD_PCA(training_maps, template, binary_mask=binary_mask, n_components=n_components)
            # X = td_pca.training_maps
            NMF.__init__(self, n_components=n_components, init='random', random_state=0, max_iter=max_iter, verbose=1)
            
            # Masking utilities
            MaskingNew.__init__(self, binary_mask, self.shape)    
            if binary_mask is not None:
                self.training_maps = self.mask(np.array(training_maps))   
                print("INFO:\tTD maps were masked.")
            else:
                self.training_maps = np.array(training_maps)

            # Normalization utilities
            Normalizations.__init__(self, self.training_maps)
    
    def fit(self, norm=None):
        return self.fit_transform_maps(norm)

    def fit_transform_maps(self, norm=None):
        """
        Fit PCA on the training data with optional normalization and return
        the transformed PCA maps.

        Parameters
        ----------
        norm : str or None, default=None
            Normalization strategy to apply before fitting PCA.
            Options are:
            - 'z_norm'   : Z-score normalization (mean 0, std 1).
            - 'min_max'  : Min-Max normalization (scales features to [0,1]).
            - 'q_norm'   : Quantile normalization.
            - None       : No normalization (raw training data).

        Returns
        -------
        transformed_maps : ndarray of shape (n_samples, n_components)
            Training maps projected into the PCA component space.

        Notes
        -----
        - The training data is provided during class initialization.
        - After applying the chosen normalization (if any), PCA is fit and the
        transformed training maps are returned.
        - Internal attributes such as `Zmaps`, `MinMaxmaps`, or `Qmaps` are
        created depending on the selected normalization.

        Raises
        ------
        NotImplementedError
            If an unsupported normalization strategy is provided.
        """
        if norm is not None:
            self.norm = norm
            if norm == 'z_norm':
                self._train_Zscore()
                print(f"INFO:\tTraining PCA with Z normalized data . . .")
                transformed_maps = self.fit_transform(self.Zmaps)
            elif norm == 'min_max':
                self._train_MinMax()
                print(f"INFO:\tTraining PCA with Min-Max normalized data . . .")
                transformed_maps = self.fit_transform(self.MinMaxmaps)
            elif norm == 'q_norm':
                self._train_QNorm()
                print(f"INFO:\tTraining PCA with Q normalized data . . .")
                transformed_maps = self.fit_transform(self.Qmaps)
            else:
                raise NotImplementedError("Choose a normalization strategy from 'z_norm', 'min_max', or 'q_norm'.")
        else:
            self.norm = None
            print(f"INFO:\tTraining PCA without normalization . . .")
            transformed_maps = self.fit_transform(self.training_maps)

        print("INFO:\tPrincipal Components (PCs) found successfully!")
        self.training_scores = transformed_maps
        return transformed_maps

    def transform_maps(self, maps):
        """
        FProject new maps into PCA space.

        Parameters
        ----------
        maps : array-like, shape (Nsubjects, Nvoxels)
            New flattened tract density maps that will
            be projected to the PCA space fitted during
            training.

        Returns
        -------
        transformed_maps : ndarray of shape (n_samples, n_components)
            Training maps projected into the PCA component space.

        Notes
        -----
        - The new data will undergo the same normalization as the training 
        tract density maps.

        Raises
        ------
        NotImplementedError
            If a the attribute 'norm' does not exist. This is because a PCA
            space has not been fit.
        """
        maps = np.array(maps)
        if hasattr(self, "mask_flat"):
            print("INFO:\tTD maps were masked.")
            maps = self.mask(maps)   

        if not hasattr(self, "norm"):
            raise RuntimeError("Must call fit_pca_maps() to apply the same normalization procedure as the training data.")

        if self.norm is not None:
            if self.norm == 'z_norm':
                print(f"INFO:\tTransforming PCA with Z normalized data . . .")
                transformed_maps = self.transform(self.Zscore(maps))
            elif self.norm == 'min_max':
                print(f"INFO:\tTransforming PCA with Min-Max normalized data . . .")
                transformed_maps = self.transform(self.MinMax(maps))
            elif self.norm == 'q_norm':
                print(f"INFO:\tTransforming PCA with Q normalized data . . .")
                transformed_maps = self.transform(self.QNorm(maps))
            else:
                pass
        else:
            print(f"INFO:\tTransforming maps without applying any normalization . . .")
            transformed_maps = self.transform(maps)

        print("INFO:\tMaps transformed sucessfully!")
        return transformed_maps
    
    def __str__(self):
        return f"NMF"
    
    def __repr__(self):
        return f"NMF"

    @property
    def value(self):
        return "NMF"