import numpy as np

class MaskingNew():
    def __init__(self, binary_mask, dims=None) -> None:
        if binary_mask is not None:
            if hasattr(binary_mask, 'get_fdata'):
                self.binary_mask = binary_mask.get_fdata()
            else:
                self.binary_mask = binary_mask
        else:
            self.binary_mask = None

        self.dims = dims
        
        # Added: Automatic creation of "mask_flat"
        if self.binary_mask is not None:
            self.mask_flat = np.asarray(self.binary_mask, dtype=bool).flatten()
        else:
            self.mask_flat = None

    def mask(self, maps):
        if self.mask_flat is None:
            raise RuntimeError("Maska nie została poprawnie zainicjalizowana.")
            
        maps = np.asarray(maps)
        if maps.ndim != 2:
            raise ValueError("Input maps must be 2D (n_samples x n_voxels).")
        
        if maps.shape[1] != len(self.mask_flat):
             raise ValueError("Liczba vokseli w danych nie zgadza się z rozmiarem maski.")

        return maps[:, self.mask_flat] 
        
    def unmask(self, data_flat, make_nan=False):
        if self.mask_flat is None:
            raise AttributeError("'mask_flat' is not defined. Check your binary_mask input.")

        n_voxels = len(self.mask_flat)
        data_flat = np.asarray(data_flat)
        
        if data_flat.ndim == 1:
            full_data = np.zeros(n_voxels)
            if make_nan: full_data[:] = np.nan
            full_data[self.mask_flat] = data_flat
        else:
            n_comp = data_flat.shape[0]
            full_data = np.zeros((n_comp, n_voxels))
            if make_nan: full_data[:] = np.nan
            
            full_data[:, self.mask_flat] = data_flat
                
        return full_data