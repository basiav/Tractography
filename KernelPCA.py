from sklearn.decomposition import KernelPCA

from normalize import Normalizations
# from masking import Masking
from masking_new import MaskingNew

import numpy as np
import matplotlib.pyplot as plt
import os

class TD_Kernel_PCA(KernelPCA, Normalizations, MaskingNew):
    def __init__(self, training_maps, template, binary_mask=None, n_components=None, kernel="rbf", gamma = 0.01) -> None:
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

        self.kernel = kernel

        # Sci-kit learn PCA method
        # PCA.__init__(self, n_components=n_components)
        KernelPCA.__init__(self, 
        n_components=n_components, kernel=kernel, gamma=gamma, fit_inverse_transform=True, alpha=0.1
        )
         
        # Masking utilities
        MaskingNew.__init__(self, binary_mask, self.shape)    
        if binary_mask is not None:
            self.training_maps = self.mask(np.array(training_maps))   
            print("INFO:\tTD maps were masked.")
        else:
            self.training_maps = np.array(training_maps)

        # Normalization utilities
        Normalizations.__init__(self, self.training_maps)   

    def fit_pca_maps(self, norm=None):
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
    
    def reconstruct_LowRank(self, transformed_data=None, n_components=None, mask_negative=True, make_nan=False):
        """
        Reconstruct data from a reduced set of PCA components (low-rank approximation).

        Parameters
        ----------
        transformed_data : array-like of shape (n_samples, n_components_used), optional
            Data projected into PCA space (scores). 
            If None, the stored training projections (`self.training_scores`) are used.
        n_components : int, optional, default=None
            Number of components to use for reconstruction.
            - If None or -1, all fitted components (`self.n_components_`) are used.
            - If smaller than the available components, a low-rank reconstruction is performed.
        mask_negative : bool, optional, default=None
            Due to the nature of PCA, reconstructions can yield negative tract densities,
            especially in voxels with original zero values. Set to True to artificially 
            clip negative values to 0.
        make_nan : bool
            Sets the voxels outside of the masked areas NaNs instead of zeros.

        Returns
        -------
        X_reconstructed : array-like of shape (n_samples, n_features)
            The reconstructed data, optionally inverse-normalized and unmasked 
            if a mask was applied during preprocessing.

        Notes
        -----
        - Reconstruction is computed as:
            X_reconstructed_norm = Z[:, :k] @ W[:k, :]
        where `Z` are the PCA scores (transformed data) and `W` are the PCA components.
        - If data were normalized before PCA (z-score, min-max, quantile), 
        the inverse normalization is applied.
        - If a mask was used, the reconstruction is unmasked to the original voxel/grid shape.
        """
        if not hasattr(self, "components_"):
            raise RuntimeError("Must call fit_pca_maps() prior to reconstruct new data.")
        
        if n_components is None or n_components==-1:
            n_components = self.n_components_

        if transformed_data is None:
            transformed_data = self.training_scores
            print(f"INFO:\tPerforming a low-rank reconstruction of the training data with {n_components} components")
        else:
            print(f"INFO:\tPerforming a low-rank reconstruction with {n_components} components")
          
        X_reconstructed_norm = transformed_data[:, :n_components] @ self.components_[:n_components, :]
        if self.norm == 'z_norm':
            print("INFO:\tApplying the inverse z_norm.")
            X_reconstructed = X_reconstructed_norm * self.sigma_train + self.mu_train
        elif self.norm == 'min_max':
            print("INFO:\tApplying the inverse min_max.")
            X_reconstructed = X_reconstructed_norm * (self.max_train - self.min_train) + self.min_train
        elif self.norm == 'q_norm':
            print("INFO:\tApplying the inverse q_norm.")
            X_reconstructed = self.qt.inverse_transform(X_reconstructed_norm)
        else:
            X_reconstructed = X_reconstructed_norm

        if hasattr(self, "mask_flat"):
            print("INFO:\tUnmasking the low-rank reconstruction.")
            X_reconstructed = self.unmask(X_reconstructed, make_nan=make_nan)
        
        if mask_negative is True:
            X_reconstructed[X_reconstructed<0] = 0
        
        return X_reconstructed

    def explained_variance_plot(self, show=False, save_dir=None, filename="explained_variance.png", step=10, dpi=100):
        """
        Plot explained variance ratio and cumulative variance with vertical lines at 50%, 80%, 95% cumulative variance.
        Cumulative variance is plotted on a second y-axis.

        Parameters
        ----------
        show : bool
            Whether to display the plot.
        save_dir : str or None
            Directory path to save the plot. If None, no file is saved.
        filename : str
            Filename for the saved plot.

        Returns
        -------
        thresholds_indices : dict
            Dictionary with keys '50%', '80%', '95%' and values as component indices.

        Raises
        ------
        RuntimeError
            If this method is called before fit_pca_maps.
        """

        if not hasattr(self, "explained_variance_ratio_") :
            raise RuntimeError("Must call fit_pca_maps() to get an explained variance percentage.")
        
        exp_var = self.explained_variance_ratio_ * 100
        cum_var = np.cumsum(exp_var)

        fig, ax1 = plt.subplots(figsize=(6,4))
        ax2 = ax1.twinx()

        # Bar plot for explained variance
        ax1.plot(range(1, len(exp_var)+1), exp_var, '-s', color='blue', markersize=2.5)
        ax1.set_xlabel("Principal components")
        ax1.set_ylabel("Explained Variance (%)")
        ax1.spines[["top"]].set_visible(False)
        ax1.set_xlim([0, len(exp_var)+1])
        ax1.set_ylim([0, exp_var[0]+5])

        # Line plot for cumulative variance on secondary axis
        ax2.plot(range(1, len(cum_var)+1), cum_var, marker='o', color='teal', markersize=2.5)
        ax2.set_ylabel("Cumulative Variance (%)")
        ax2.set_yticks([0,20,40,60,80,100])
        ax2.set_ylim([0, 110])
        ax2.spines[["top"]].set_visible(False)
        ax2.spines['right'].set_color('teal')
        ax2.tick_params(axis='y', colors='teal')

        # Add vertical lines for 50%, 80%, 95% cumulative variance
        thresholds = [50, 80, 95]
        styles = ["-", "--", "dotted"]
        thresholds_indices = {}
        for t, s in zip(thresholds, styles):
            idx = np.searchsorted(cum_var, t) + 1
            thresholds_indices[f"{t}%"] = idx
            ax1.vlines(x=idx, ymin=0, ymax=exp_var[0], color="gray", linestyle=s, label=f"{t}% at #{idx}")

        # Combine legends
        lines_labels = [ax.get_legend_handles_labels() for ax in [ax1, ax2]]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        ax1.legend(lines, labels, loc='upper center', ncols=3, frameon=False)

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, filename)
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close(fig)

        return thresholds_indices
    

    def get_pca_maps(self):
        """
        Wyciąga mapy przestrzenne z Kernel PCA.
        """
        if not hasattr(self, "X_transformed_fit_"):
            raise RuntimeError("Najpierw wywołaj fit_pca_maps()!")

        # Sprawdzamy, ile komponentów faktycznie zostało dopasowanych
        # To jest kluczowe dla Kernel PCA
        actual_n_components = self.X_transformed_fit_.shape[1]
        
        print(f"INFO: Model dopasował {actual_n_components} komponentów.")
        
        # Tworzymy macierz jednostkową o poprawnym wymiarze
        identity = np.eye(actual_n_components)
        
        # Teraz inverse_transform nie wyrzuci błędu wymiarów
        raw_components = self.inverse_transform(identity)
        
        # Unmasking, aby uzyskać kształt 3D/4D (98, 116, 94, actual_n_components)
        if hasattr(self, "unmask"):
            # Upewnij się, że Twoja metoda unmask przyjmuje dane o kształcie (n_comp, n_voxels)
            return self.unmask(raw_components)
        
        return raw_components

    def __str__(self):
        return "KernelPCA"
    
    def __repr__(self):
        return "KernelPCA"
    
    @property
    def value(self):
        return self.kernel