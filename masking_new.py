# import numpy as np

# class MaskingNew():
#     def __init__(self, binary_mask, dims=None) -> None:
#         if binary_mask is not None:
#             if hasattr(binary_mask, 'get_fdata'):
#                 self.binary_mask = binary_mask.get_fdata()
#             else:
#                 self.binary_mask = binary_mask
#         else:
#             self.binary_mask = None

#         self.dims = dims
        
#         # Added: Automatic creation of "mask_flat"
#         if self.binary_mask is not None:
#             self.mask_flat = np.asarray(self.binary_mask, dtype=bool).flatten()
#         else:
#             self.mask_flat = None

#     def mask(self, maps):
#         if self.mask_flat is None:
#             raise RuntimeError("Maska nie została poprawnie zainicjalizowana.")
            
#         maps = np.asarray(maps)
#         if maps.ndim != 2:
#             raise ValueError("Input maps must be 2D (n_samples x n_voxels).")
        
#         if maps.shape[1] != len(self.mask_flat):
#              raise ValueError("Liczba vokseli w danych nie zgadza się z rozmiarem maski.")

#         return maps[:, self.mask_flat] 
        
#     def unmask(self, data_flat, make_nan=False):
#         if self.mask_flat is None:
#             raise AttributeError("'mask_flat' is not defined. Check your binary_mask input.")

#         n_voxels = len(self.mask_flat)
#         data_flat = np.asarray(data_flat)
        
#         if data_flat.ndim == 1:
#             full_data = np.zeros(n_voxels)
#             if make_nan: full_data[:] = np.nan
#             full_data[self.mask_flat] = data_flat
#         else:
#             n_comp = data_flat.shape[0]
#             full_data = np.zeros((n_comp, n_voxels))
#             if make_nan: full_data[:] = np.nan
            
#             full_data[:, self.mask_flat] = data_flat
                
#         return full_data

import numpy as np

class MaskingNew():
    def __init__(self, binary_mask, dims=None) -> None:
        if binary_mask is not None:
            if hasattr(binary_mask, 'get_fdata'):
                self.binary_mask = binary_mask.get_fdata()
            else:
                self.binary_mask = np.asarray(binary_mask)
        else:
            self.binary_mask = None

        self.dims = dims
        
        # Płaska maska boolowska
        if self.binary_mask is not None:
            self.mask_flat = np.asarray(self.binary_mask, dtype=bool).flatten()
        else:
            self.mask_flat = None

    def mask(self, maps):
        if self.mask_flat is None:
            raise RuntimeError("Maska nie została poprawnie zainicjalizowana.")
            
        maps = np.asarray(maps)
        if maps.ndim != 2:
            raise ValueError(f"Input maps must be 2D (n_samples x n_voxels). Got {maps.ndim}D instead.")
        
        if maps.shape[1] != len(self.mask_flat):
             raise ValueError(f"Liczba wokseli w danych ({maps.shape[1]}) nie zgadza się z całkowitym rozmiarem spłaszczonej maski ({len(self.mask_flat)}).")

        return maps[:, self.mask_flat] 
        
    def unmask(self, data_flat, make_nan=False):
        if self.mask_flat is None:
            raise AttributeError("'mask_flat' is not defined. Check your binary_mask input.")

        n_voxels_total = len(self.mask_flat)
        data_flat = np.asarray(data_flat)
        
        if data_flat.ndim == 1:
            # Dla pojedynczego wektora (np. 1 mapy komponentu)
            full_data = np.zeros(n_voxels_total)
            if make_nan: 
                full_data[:] = np.nan
            full_data[self.mask_flat] = data_flat
            
            # Automatyczny reshape do 3D jeśli podano wymiary w konstruktorze!
            if self.dims is not None:
                full_data = full_data.reshape(self.dims)
                
        else:
            # Dla macierzy 2D (np. n_components x woksele_w_masce)
            n_comp = data_flat.shape[0]
            
            # W neuroobrazowaniu zazwyczaj chcemy trzymać kształt: (woksele_total, n_comp)
            # lub zachować oryginał: (n_comp, woksele_total).
            # Zachowamy strukturę Scikit-Learn: (n_comp, n_voxels)
            full_data = np.zeros((n_comp, n_voxels_total))
            if make_nan: 
                full_data[:] = np.nan
            
            full_data[:, self.mask_flat] = data_flat
            
            # Jeśli chcemy bezpośrednio tworzyć pliki Nifti 4D:
            if self.dims is not None:
                # Najpierw zmieniamy na (n_voxels, n_comp), żeby reshape 3D poszło łatwiej
                # wynikiem będzie (X, Y, Z, n_comp)
                full_data = full_data.T.reshape(self.dims + (n_comp,))
                
        return full_data