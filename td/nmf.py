# from sklearn.decomposition import NMF

# from normalize import Normalizations
# # from masking import Masking
# from masking_new import MaskingNew

# import numpy as np
# import matplotlib.pyplot as plt
# import os

# from TractDensityComponents import TD_PCA


# class NMFProjections(MaskingNew, NMF):
#     def __init__(self, training_maps, template, binary_mask=None, n_components=None, max_iter: int = 10) -> None:
#             """
#             training_maps : array-like, shape (Nsubjects, Nvoxels)
#                 Training dataset (flattened tract density maps).
#             template : nibabel.Nifti1Image
#                 Reference image providing the affine and header information.
#             binary_mask : nibabel.Nifti1Image
#                 Binary mask with the voxels to use for the PCA fit.
#             n_components : int or None
#                 Number of PCA components to retain.
#             """

#             self.shape = template.get_fdata().shape
#             self.template = template

#             # To mask data
#             # td_pca = TD_PCA(training_maps, template, binary_mask=binary_mask, n_components=n_components)
#             # X = td_pca.training_maps
#             NMF.__init__(self, n_components=n_components, init='random', random_state=0, max_iter=max_iter, verbose=1)
            
#             # Masking utilities
#             MaskingNew.__init__(self, binary_mask, self.shape)    
#             if binary_mask is not None:
#                 # Use asarray and cast to float32
#                 self.training_maps = self.mask(np.asarray(training_maps, dtype=np.float32))   
#                 print("INFO:\tTD maps were masked.")
#             else:
#                 self.training_maps = np.asarray(training_maps, dtype=np.float32)

#             # Normalization utilities
#             Normalizations.__init__(self, self.training_maps)
    
#     def fit(self, norm=None):
#         return self.fit_transform_maps(norm)

#     def fit_transform_maps(self, norm=None):
#         """
#         Fit PCA on the training data with optional normalization and return
#         the transformed PCA maps.

#         Parameters
#         ----------
#         norm : str or None, default=None
#             Normalization strategy to apply before fitting PCA.
#             Options are:
#             - 'z_norm'   : Z-score normalization (mean 0, std 1).
#             - 'min_max'  : Min-Max normalization (scales features to [0,1]).
#             - 'q_norm'   : Quantile normalization.
#             - None       : No normalization (raw training data).

#         Returns
#         -------
#         transformed_maps : ndarray of shape (n_samples, n_components)
#             Training maps projected into the PCA component space.

#         Notes
#         -----
#         - The training data is provided during class initialization.
#         - After applying the chosen normalization (if any), PCA is fit and the
#         transformed training maps are returned.
#         - Internal attributes such as `Zmaps`, `MinMaxmaps`, or `Qmaps` are
#         created depending on the selected normalization.

#         Raises
#         ------
#         NotImplementedError
#             If an unsupported normalization strategy is provided.
#         """
#         if norm is not None:
#             self.norm = norm
#             if norm == 'z_norm':
#                 self._train_Zscore()
#                 print(f"INFO:\tTraining PCA with Z normalized data . . .")
#                 transformed_maps = self.fit_transform(self.Zmaps)
#             elif norm == 'min_max':
#                 self._train_MinMax()
#                 print(f"INFO:\tTraining PCA with Min-Max normalized data . . .")
#                 transformed_maps = self.fit_transform(self.MinMaxmaps)
#             elif norm == 'q_norm':
#                 self._train_QNorm()
#                 print(f"INFO:\tTraining PCA with Q normalized data . . .")
#                 transformed_maps = self.fit_transform(self.Qmaps)
#             else:
#                 raise NotImplementedError("Choose a normalization strategy from 'z_norm', 'min_max', or 'q_norm'.")
#         else:
#             self.norm = None
#             print(f"INFO:\tTraining PCA without normalization . . .")
#             transformed_maps = self.fit_transform(self.training_maps)

#         print("INFO:\tPrincipal Components (PCs) found successfully!")
#         self.training_scores = transformed_maps
#         return transformed_maps

#     def transform_maps(self, maps):
#         """
#         FProject new maps into PCA space.

#         Parameters
#         ----------
#         maps : array-like, shape (Nsubjects, Nvoxels)
#             New flattened tract density maps that will
#             be projected to the PCA space fitted during
#             training.

#         Returns
#         -------
#         transformed_maps : ndarray of shape (n_samples, n_components)
#             Training maps projected into the PCA component space.

#         Notes
#         -----
#         - The new data will undergo the same normalization as the training 
#         tract density maps.

#         Raises
#         ------
#         NotImplementedError
#             If a the attribute 'norm' does not exist. This is because a PCA
#             space has not been fit.
#         """
#         maps = np.asarray(maps, dtype=np.float32)
        
#         if hasattr(self, "mask_flat"):
#             print("INFO:\tTD maps were masked.")
#             maps = self.mask(maps)

#         if not hasattr(self, "norm"):
#             raise RuntimeError("Must call fit_pca_maps() to apply the same normalization procedure as the training data.")

#         if self.norm is not None:
#             if self.norm == 'z_norm':
#                 print(f"INFO:\tTransforming PCA with Z normalized data . . .")
#                 transformed_maps = self.transform(self.Zscore(maps))
#             elif self.norm == 'min_max':
#                 print(f"INFO:\tTransforming PCA with Min-Max normalized data . . .")
#                 transformed_maps = self.transform(self.MinMax(maps))
#             elif self.norm == 'q_norm':
#                 print(f"INFO:\tTransforming PCA with Q normalized data . . .")
#                 transformed_maps = self.transform(self.QNorm(maps))
#             else:
#                 pass
#         else:
#             print(f"INFO:\tTransforming maps without applying any normalization . . .")
#             transformed_maps = self.transform(maps)

#         print("INFO:\tMaps transformed sucessfully!")
#         return transformed_maps
    
#     def __str__(self):
#         return f"NMF"
    
#     def __repr__(self):
#         return f"NMF"

#     @property
#     def value(self):
#         return "NMF"
    
#     def get_feature_names_out(self, input_features=None):
#         import numpy as np
#         return np.array([f"NMF_Comp_{i}" for i in range(1, self.n_components_ + 1)])

import numpy as np
import os
from sklearn.decomposition import NMF

from normalize import Normalizations
from masking_new import MaskingNew

# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"


class NMFProjections(MaskingNew, NMF):
    def __init__(self, training_maps, template, binary_mask=None, n_components=None, max_iter: int = 10) -> None:
        """
        NMF Projection class for Tract Density Maps optimized for memory footprint.

        training_maps : array-like, shape (Nsubjects, Nvoxels)
            Training dataset (flattened tract density maps).
        template : nibabel.Nifti1Image
            Reference image providing the affine and header information.
        binary_mask : nibabel.Nifti1Image
            Binary mask with the voxels to use for the NMF fit.
        n_components : int or None
            Number of NMF components to retain.
        max_iter : int, default=10
            Maximum number of iterations for NMF solver.
        """
        self.shape = template.get_fdata().shape
        self.template = template

        # ZMIANA: Zmieniono init z 'random' na 'nndsvdar' - zapewnia znacznie lepszą
        # zbieżność dla danych nieujemnych (obrazy) i oszczędza RAM (mniej iteracji).
        NMF.__init__(
            self, 
            n_components=n_components, 
            init='nndsvdar', 
            random_state=0, 
            max_iter=max_iter, 
            verbose=1
        )
        
        # Maskowanie
        MaskingNew.__init__(self, binary_mask, self.shape)    
        if binary_mask is not None:
            self.training_maps = self.mask(np.asarray(training_maps, dtype=np.float32))   
            print("INFO:\tTD maps were masked.")
        else:
            self.training_maps = np.asarray(training_maps, dtype=np.float32)

        # KLUCZOWA ZMIANA: Usunięto Normalizations.__init__(self, self.training_maps)
        # Nie odpalamy całego konstruktora normalizacji na starcie, żeby nie generować
        # trzech kopii gigantycznych tablic (Zmaps, MinMaxmaps, Qmaps) w RAM-ie.
        self.norm = None

    def fit(self, norm=None):
        return self.fit_transform_maps(norm)

    def fit_transform_maps(self, norm=None):
        """
        Fit NMF on the training data with optional normalization and return
        the transformed NMF scores.

        Parameters
        ----------
        norm : str or None, default=None
            Normalization strategy to apply before fitting NMF.
            Options are: 'z_norm', 'min_max', 'q_norm', None.
        """
        self.norm = norm
        
        if norm is not None:
            # Inicjalizujemy obiekt lokalnie, tylko w momencie wykonywania fitowania
            normalizer = Normalizations(self.training_maps)
            
            if norm == 'z_norm':
                print("INFO:\tTraining NMF with Z normalized data . . .")
                # WAŻNE: NMF nie akceptuje wartości ujemnych! Jeśli Z-norm schodzi poniżej zera,
                # upewnij się, że Twoja klasa Normalizations radzi sobie z tym problemem.
                data_to_fit = normalizer.Zmaps
            elif norm == 'min_max':
                print("INFO:\tTraining NMF with Min-Max normalized data . . .")
                data_to_fit = normalizer.MinMaxmaps
            elif norm == 'q_norm':
                print("INFO:\tTraining NMF with Q normalized data . . .")
                data_to_fit = normalizer.Qmaps
            else:
                raise NotImplementedError("Choose a normalization strategy from 'z_norm', 'min_max', or 'q_norm'.")
        else:
            print("INFO:\tTraining NMF without normalization . . .")
            data_to_fit = self.training_maps

        # Wykonanie dekompozycji
        transformed_maps = self.fit_transform(data_to_fit)

        # CZYSZCZENIE RAMU: Natychmiast usuwamy obiekt normalizujący i zwalniamy pamięć
        if norm is not None:
            del normalizer

        print("INFO:\tNon-negative Matrix Factorization (NMF) completed successfully!")
        self.training_scores = transformed_maps
        return transformed_maps

    def transform_maps(self, maps):
        """
        Project new maps into the fitted NMF space.
        """
        maps = np.asarray(maps, dtype=np.float32)
        
        if hasattr(self, "mask_flat"):
            print("INFO:\tTD maps were masked.")
            maps = self.mask(maps)

        if not hasattr(self, "norm"):
            raise RuntimeError("Must call fit_transform_maps() to apply the same normalization procedure as the training data.")

        # Do transformacji potrzebujemy na chwilę metod instancyjnych z Normalizations
        if self.norm is not None:
            # Tworzymy lekki obiekt "dummy" do transformacji nowych danych
            normalizer = Normalizations(self.training_maps)
            
            if self.norm == 'z_norm':
                print("INFO:\tTransforming maps with Z normalized data . . .")
                transformed_maps = self.transform(normalizer.Zscore(maps))
            elif self.norm == 'min_max':
                print("INFO:\tTransforming maps with Min-Max normalized data . . .")
                transformed_maps = self.transform(normalizer.MinMax(maps))
            elif self.norm == 'q_norm':
                print("INFO:\tTransforming maps with Q normalized data . . .")
                transformed_maps = self.transform(normalizer.QNorm(maps))
            
            del normalizer  # Czyszczenie pamięci
        else:
            print("INFO:\tTransforming maps without applying any normalization . . .")
            transformed_maps = self.transform(maps)

        print("INFO:\tMaps transformed successfully!")
        return transformed_maps
    
    def __str__(self):
        return "NMF"
    
    def __repr__(self):
        return "NMF"

    @property
    def value(self):
        return "NMF"
    
    def get_feature_names_out(self, input_features=None):
        n_comps = self.n_components if self.n_components is not None else self.n_components_
        return np.array([f"NMF_Comp_{i}" for i in range(1, n_comps + 1)])