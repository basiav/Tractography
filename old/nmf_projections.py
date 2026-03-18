# TODO: Add the low-rank reconstructions
# TODO: Add a way to measure and assess the scores (i.e., contribution of each component to the TD map)

# TODO: Function to compare components computed from different datasets
# TODO: Does it make sense to somehow consider white matter atlasses? So we can interpret which component involves what? What do negative values mean in this case?

import warnings
import os
import numpy as np
import nibabel as nib
from nilearn import plotting, datasets, surface, plotting, image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import itertools
from joblib import Parallel, delayed

from utils import Resample_ColorMap
from masking import Masking

import os
import numpy as np
import nibabel as nib
from nilearn import image
from masking_new import MaskingNew


class NMFProjections(MaskingNew):
    def __init__(self, W, H, template, binary_mask):
        """
        W: Macierz wag (pacjenci x komponenty)
        H: Macierz komponentów (komponenty x voksele)
        template: Obraz NIfTI (jako wzorzec afiny i nagłówka)
        binary_mask: Obraz NIfTI lub macierz maski
        """
        self.W = W
        self.components = H
        self.n_components_ = W.shape[1]
        
        if hasattr(template, 'get_fdata'):
            self.shape = template.get_fdata().shape
        else:
            self.shape = template.shape
            
        self.template = template

        # 1. Inicjalizacja Masking - to stworzy self.mask_flat
        super().__init__(binary_mask, self.shape) 

        # 2. Odwijamy komponenty z formy płaskiej do 4D
        self._unflatten_components(self.components)

    def _unflatten_components(self, components):
        """
        Konwertuje płaskie komponenty H na objętość 4D (X, Y, Z, n_komponentów).
        """

        print(f"DEBUG: components shape: {components.shape}")
        print(f"DEBUG: n_components_: {self.n_components_}")
        print(f"DEBUG: expected total size: {self.n_components_ * np.prod(self.shape)}")
        # Używamy metody unmask z klasy bazowej
        print("INFO:\tUnmasking NMF loadings into 3D space...")
        # Wynik: (n_components, total_voxels_in_box)
        unmasked_data = self.unmask(components, make_nan=False)

        # Reshape do 4D: (n_components, X, Y, Z)
        comp_4d_shape = (self.n_components_,) + self.shape
        comp_4d = unmasked_data.reshape(comp_4d_shape)

        # Reorder do standardu NIfTI: (X, Y, Z, n_components)
        self._components_4d = np.transpose(comp_4d, (1, 2, 3, 0))

    def components_to_brain(self, save_dir=None, filename=None, verbose=True):
        """
        Zapisuje komponenty jako plik NIfTI 4D.
        """
        comp_img = nib.Nifti1Image(
            self._components_4d, 
            affine=self.template.affine, 
            header=self.template.header
        )

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, filename)
            nib.save(comp_img, save_path)
            if verbose:
                print(f"INFO:\tSaved 4D NIfTI components at {save_path}")
        
        return comp_img


# class NMFProjections(Masking):
#     def __init__(self, W, H, template, binary_mask):
#         self.W = W
#         self.components = H
#         self.n_components_ = W.shape[1]
#         if W.shape[1] != H.shape[0]:
#             raise ValueError(f"W.shape[1] != H.shape[0]: {W.shape}, {H.shape}")
#         # Pobieramy dane maski jako tablicę numpy (ważne!)
#         mask_data = binary_mask.get_fdata() if hasattr(binary_mask, 'get_fdata') else binary_mask
#         self.shape = mask_data.shape # Pobierz kształt z maski
#         self.template = template

#         # Inicjalizacja klasy nadrzędnej Masking
#         Masking.__init__(self, mask_data, self.shape) 

#         if binary_mask is not None:
#             if hasattr(binary_mask, 'get_fdata'):
#                 self.binary_mask = binary_mask.get_fdata()
#             else:
#                 self.binary_mask = binary_mask # To już jest macierz, nie trzeba get_fdata()
#         else:
#             self.binary_mask = None
        
#         # TO JEST KLUCZOWE: Ustawienie mask_flat, aby unmask zadziałało
#         # self.mask_flat = mask_data.flatten()
#         # self.binary_mask = mask_data

#         # Teraz unmask zadziała i uzupełni brakujące voksele zerami/NaNami
#         self._unflatten_components(self.components)

#     def _unflatten_components(self, components):
#         """
#         Convert flattened PCA components into a 4D volume (X, Y, Z, n_components).
#         """
#         if hasattr(self, "mask_flat"):
#             print("INFO:\tUnmasking PCA loadings.")
#             components = self.unmask(components, make_nan=True)
#             mask_flat = self.binary_mask.flatten()
#             self.mask_flat = mask_flat 

#         comp_4d = components.reshape((self.n_components_,) + self.shape)
#         self._components_4d = np.transpose(comp_4d, (1, 2, 3, 0))  # reorder to (X, Y, Z, n_components)

#     def components_to_brain(self, save_dir=None, filename=None, verbose=True):
#         if save_dir is not None:
#             os.makedirs(save_dir, exist_ok=True)
#             save_path = os.path.join(save_dir, filename)

#             comp_img = nib.Nifti1Image(self._components_4d, affine=self.template.affine, header=self.template.header)
#             nib.save(comp_img, save_path)
#             if verbose:
#                 print(f"INFO:\tSaved 4D NIfTI PCA components at {save_path}")
#         else:
#             return self._components_4d
    