import numpy as np

class Masking():
    def __init__(self, binary_mask, dims=None) -> None:
        """
        Class for applying and managing a binary mask for voxel-based data.

        Parameters
        ----------
        binary_mask : nibabel.Nifti1Image
            Binary mask (1 = keep voxel, 0 = ignore voxel).
            This defines which voxels are included in the analysis.
        dims : tuple of int, optional
            Original dimensions of the template image (e.g., (X, Y, Z)).
            If None, the shape is inferred from `binary_mask`.

        Attributes
        ----------
        binary_mask : ndarray
            The input binary mask.
        dims : tuple of int
            Dimensions of the original template volume.
        mask_flat : ndarray of shape (X*Y*Z,)
            Flattened version of the mask (boolean), useful for indexing.

        Notes
        -----
        - The mask can be used to reduce the dimensionality of voxelwise
          data by discarding non-brain voxels (or irrelevant regions).
        - Use this together with PCA or other dimensionality reduction methods
          to avoid processing meaningless zero-valued voxels.
        """
        # BEFORE:
        # self.binary_mask = binary_mask.get_fdata() if binary_mask is not None else None

        # AFTER:
        if binary_mask is not None:
            if hasattr(binary_mask, 'get_fdata'):
                self.binary_mask = binary_mask.get_fdata()
            else:
                self.binary_mask = binary_mask
        else:
            self.binary_mask = None

        self.dims = dims

    def mask(self, maps):
        """
        Apply a binary mask to flattened data, keeping only voxels inside the mask.

        Parameters
        ----------
        maps : ndarray of shape (n_samples, n_voxels), optional
            Flattened maps to mask.

        Returns
        -------
        masked_maps : ndarray
            Masked maps containing only voxels inside the mask.

        Raises
        ------
        ValueError
            If input maps are not 2D (n_samples x n_voxels).
            If binary_mask does not match the stored template shape.

        Notes
        -----
        - Stores mask information in `self.mask_flat` for later unflattening.
        - Stores `self.template_shape` from the binary mask for reconstruction.
        """
        self.binary_mask = np.asarray(self.binary_mask, dtype=bool)
        if hasattr(self, "dims"):
            assert self.binary_mask.shape == self.dims, "Binary mask and tempalte dimensions do not match" 

        # Flatten mask
        mask_flat = self.binary_mask.flatten()
        self.mask_flat = mask_flat  # save for reference

        maps = np.asarray(maps)
        if maps.ndim != 2:
            raise ValueError("Input maps must be 2D (n_samples x n_voxels).")
        return maps[:, mask_flat] 
        
    def unmask(self, data_flat, make_nan=False):
        """
        Reconstruct full 2D or 3D maps from masked data by inserting zeros
        in voxels outside the mask.

        Parameters
        ----------
        masked_maps : ndarray of shape (n_samples, n_masked_voxels)
            Masked maps to reconstruct.
        make_nan : bool
            Sets the voxels outside of the masked areas NaNs instead of zeros.

        Returns
        -------
        full_maps : ndarray of shape (n_samples, n_voxels)
            Reconstructed maps with zeros in voxels outside the mask.

        Raises
        ------
        RuntimeError
            If `self.mask_flat` is not defined.
        """
        if not hasattr(self, "mask_flat"):
            raise RuntimeError("Mask information not found. Apply `mask()` first.")

        masked_maps = np.asarray(masked_maps)
        n_samples = masked_maps.shape[0]
        n_voxels = self.mask_flat.size

        if masked_maps.shape[1] != self.mask_flat.sum():
            raise ValueError(f"Number of columns in masked_maps ({masked_maps.shape[1]}) "
                            f"does not match number of masked voxels ({self.mask_flat.sum()}).")

        # Initialize full maps with zeros
        full_maps = np.zeros((n_samples, n_voxels), dtype=masked_maps.dtype)
        if make_nan:
            full_maps = full_maps * np.nan

        # Fill in masked voxels
        full_maps[:, self.mask_flat] = masked_maps  

        return full_maps