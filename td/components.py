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

class Projections:
    def __init__(self, TD_PCA):
        """
        Initialize a Projections object for visualizing and saving PCA components in voxel space.

        Parameters
        ----------
        TD_PCA : TD_PCA
            A fitted TD_PCA instance (must contain PCA components).

        Raises
        ------
        Warning
            If the provided TD_PCA object has not been fitted yet.
        """
        self.TD_PCA = TD_PCA

        # Check if PCA has been fit
        if not hasattr(TD_PCA, "components_"):
            warnings.warn("The provided TD_PCA object has not been fitted yet. "
                          "Call `fit_pca` or `fit_transform_pca` first.")
        else:
            # Store key attributes from the fitted PCA
            self.components_ = TD_PCA.components_
            self.n_components_ = TD_PCA.n_components_
            self.exp_var = TD_PCA.explained_variance_ratio_ * 100
            self.shape = TD_PCA.shape
            self.template = TD_PCA.template

        # Masking info (optional)
        if hasattr(TD_PCA, "mask_flat"):
            self.mask_flat = TD_PCA.mask_flat
        else:
            self.mask_flat = None

        # Cache the 4D components in voxel space
        self._unflatten_components(TD_PCA.components_)

    def _unflatten_components(self, components):
        """
        Convert flattened PCA components into a 4D volume (X, Y, Z, n_components).
        """
        if hasattr(self.TD_PCA, "mask_flat"):
            print("INFO:\tUnmasking PCA loadings.")
            components = self.TD_PCA.unmask(components, make_nan=True)

        comp_4d = components.reshape((self.n_components_,) + self.shape)
        self._components_4d = np.transpose(comp_4d, (1, 2, 3, 0))  # reorder to (X, Y, Z, n_components)

    def components_to_brain(self, save_dir=None, filename=None, verbose=True):
        """
        Save all PCA component loadings as a single 4D NIfTI image.

        The resulting image has shape (X, Y, Z, n_components), where each
        volume along the fourth axis corresponds to a PCA component.

        Parameters
        ----------
        save_dir : str or None
            Directory where the NIfTI file will be saved. If None, returns the array.
        filename : str or None
            Filename for the saved NIfTI image.
        verbose : bool, optional, default=True
            If True, display information

        Returns
        -------
        comp_4d : np.ndarray or None
            If `save_dir` is None, returns the 4D array (X, Y, Z, n_components).
            Otherwise, saves the NIfTI image and returns None.
        """
        print(f"Components shape (4D): {self._components_4d.shape}")
        print(f"Template shape (3D): {self.template.shape}")
        print(f"No. of voxels in template shape: {np.prod(self.template.shape)}")

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, filename)

            full_voxels_data = self.TD_PCA.unmask(self.TD_PCA.components_)

            n_comp = self._components_4d.shape[0]

            n_comp = full_voxels_data.shape[0]
            X, Y, Z = self.template.shape[:3]

            tmp_4d = full_voxels_data.reshape(n_comp, X, Y, Z)
            comp_reshaped = tmp_4d.transpose(1, 2, 3, 0)
            
            comp_img = nib.Nifti1Image(comp_reshaped, affine=self.template.affine, header=self.template.header)
            nib.save(comp_img, save_path)
            if verbose:
                print(f"INFO:\tSaved 4D NIfTI PCA components at {save_path}")
        else:
            return self._components_4d
        
    def plot_component_volumes(
        self, 
        background_image=None,
        component_idx=0,
        save_dir=None, 
        filename=None, 
        dpi=100,
        cmap="cold_hot",
        cut_coords=None, 
        threshold=None,
        annotate=True,
        colorbar=True,
        verbose=True,
        **kwargs
    ):
        """
        Plot a PCA component in MNI space using Nilearn.

        Parameters
        ----------
        background_image : nibabel.Nifti1Image
            The background image to plot on top of. 
            If nothing is specified, the MNI152 template 
            will be used. To turn off background image, 
            just pass “bg_img=None”. Default=MNI152TEMPLATE.
        component_idx : int, default=0
            Index of the PCA component to plot.
        save_dir : str or None
            Directory where the NIfTI file will be saved. If None, returns the array.
        filename : str or None
            Filename for the saved figure.
        dpi : int or None
            Resolution of the saved figure.
        cmap : str, default="cold_hot"
            Colormap used to display the component map.
        cut_coords : None, int, or tuple, optional
            Coordinates of cuts in the plot. If None, cuts are automatically selected.
        threshold : float or None, optional
            If not None, voxels with values in [-threshold, +threshold] are set to NaN.
            Useful for suppressing low-intensity values.
        annotate : bool, default=False
            Whether to show annotations (e.g., left/right labels).
        colorbar : bool, optional, default=True
            If True, display a colorbar on the right of the plots.
        verbose : bool, optional, default=True
            If True, display information
        **kwargs :
            Additional keyword arguments passed to
            `nilearn.plotting.plot_stat_map`.

        Returns
        -------
        display : nilearn.plotting.Displays
            The display object (useful for further customization).

        Raises
        ------
        IndexError
            If component_idx is out of range.

        Notes
        -----
        - The method uses the stored template from TD_PCA for affine and header info.
        - Ensure that the template is aligned with the MNI space used in Nilearn.
        Otherwise, components may appear misaligned.
        """
        if component_idx >= self.n_components_:
            raise IndexError(
                f"Component index {component_idx} out of range "
                f"(max {self.n_components_-1})."
            )

        comp_img_4d = self._components_4d[..., component_idx]
        if threshold is not None:
            comp_img_4d = comp_img_4d.copy()
            comp_img_4d[np.abs(comp_img_4d) < threshold] = 0

        vol = nib.Nifti1Image(
            comp_img_4d, affine=self.template.affine, header=self.template.header
        )

        display = plotting.plot_stat_map(
            vol,
            bg_img=background_image,
            display_mode="mosaic",
            cut_coords=cut_coords,
            cmap=cmap,
            annotate=annotate,
            title=f"Loadings component #{component_idx+1} (Variance explained : {self.exp_var[component_idx].round(2)}%)",
            colorbar=colorbar,
            **kwargs
        )
        
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, filename)
            display.savefig(save_path, dpi=dpi)
            plt.close()
            if verbose:
                print(f"INFO:\tSaved volumetric plots to {save_path}")
        else:
            plt.show()
     
    def plot_component_surface(
        self,
        component_idx=0,
        save_dir=None, 
        filename=None, 
        dpi=100,
        threshold=None,
        figsize=(20, 4),
        colorbar=True,
        verbose=True,
        **kwargs
    ):
        """
        Project a single PCA component map to fsaverage surfaces and plot it.

        Parameters
        ----------
        component_idx : int, default=0
            Index of the PCA component to visualize.
        save_dir : str or None
            Directory where the figure will be saved. If None, the figure is just shown.
        filename : str or None
            Filename for the saved figure (used only if save_dir is provided).
        dpi : int, default=100
            Resolution of the saved figure.
        threshold : float or None, optional
            If not None, voxels with values in [-threshold, +threshold] are set to NaN.
            Useful for suppressing low-intensity values.
        figsize : tuple, default=(20, 4)
            Size of the matplotlib figure.
        colorbar : bool, optional, default=True
            If True, display a colorbar on the right of the plots.
        verbose : bool, optional, default=True
            If True, display information
        **kwargs :
            Additional keyword arguments passed to
            `nilearn.plotting.plot_surf_stat_map`.

        Notes
        -----
        - Uses fsaverage surface meshes from Nilearn.
        - Shows four views for the selected component:
            left-lateral, left-medial, right-lateral, right-medial.
        """
        # Validate index
        if component_idx >= self.n_components_:
            raise IndexError(f"Component index {component_idx} out of range (max {self.n_components_-1}).")
        
        # Fetch fsaverage surface meshes
        fsaverage = datasets.fetch_surf_fsaverage()

        # Load 4D NIfTI of PCA components
        if not hasattr(self, "_components_4d"):
            raise RuntimeError("PCA components must be available in voxel space before projection.")

        # Extract single volume
        comp_img_4d = nib.Nifti1Image(self._components_4d, affine=self.template.affine, header=self.template.header)
        vol = image.index_img(comp_img_4d, component_idx)

        # Project to surfaces
        texture_left = surface.vol_to_surf(vol, surf_mesh=fsaverage["pial_left"], inner_mesh=fsaverage["white_left"])
        texture_right = surface.vol_to_surf(vol, surf_mesh=fsaverage["pial_right"], inner_mesh=fsaverage["white_right"])

        # Create figure with 1 row × 4 columns
        fig, axes = plt.subplots(1, 4, figsize=figsize, subplot_kw={'projection': '3d'})

        # Left hemisphere lateral
        plotting.plot_surf_stat_map(
            fsaverage.infl_left, texture_left,
            hemi="left", view="lateral", colorbar=False,
            axes=axes[0], threshold=threshold, **kwargs
        )
        axes[0].set_title(f"Loadings component #{component_idx+1} - Left Lateral", fontsize=10, pad=3)

        # Left hemisphere medial
        plotting.plot_surf_stat_map(
            fsaverage.infl_left, texture_left,
            hemi="left", view="medial", colorbar=False,
            axes=axes[1], threshold=threshold, **kwargs
        )
        axes[1].set_title(f"Loadings component #{component_idx+1} - Left Medial", fontsize=10, pad=3)

        # Right hemisphere lateral
        plotting.plot_surf_stat_map(
            fsaverage.infl_right, texture_right,
            hemi="right", view="lateral", colorbar=False,
            axes=axes[2], threshold=threshold, **kwargs
        )
        axes[2].set_title(f"Loadings component #{component_idx+1} - Right Lateral", fontsize=10, pad=3)

        # Right hemisphere medial
        plotting.plot_surf_stat_map(
            fsaverage.infl_right, texture_right,
            hemi="right", view="medial", colorbar=colorbar,
            axes=axes[3], threshold=threshold, **kwargs
        )
        axes[3].set_title(f"Loadings component #{component_idx+1} - Right Medial", fontsize=10, pad=3)

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, filename)
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
            plt.close()
            if verbose:
                print(f"INFO:\tSaved surface plots to {save_path}")
        else:
            plt.show()

    def inspect_scores(
        self,
        components,
        overall_survival,
        comp_maps=None, 
        censorship=None,
        save_dir=None, 
        filename=None,
        cmap="viridis", 
        figsize=(16, 12),
        dpi=100, 
        share_colorbar=False, 
        bbox_to_anchor=(.95, 0.05),
        p=90,
        clabel="Overall Survival (OS)",
        verbose=True
    ):
        """
        Plot scatterplots of PCA component pairs, coloring uncensored patients by
        overall survival (OS) and showing censored patients in light gray.

        This function creates pairwise scatterplots between the given components.
        Each point corresponds to one patient. Patients with an event (uncensored)
        are colored according to OS, while censored patients are shown in gray.
        Optionally, a shared colorbar can be used across all subplots.

        Parameters
        ----------
        components : list of int
            List of component indices to include in pairwise plots (e.g., [0,1,2,3]).
            All unique pairs will be plotted. 
        overall_survival : array-like, shape (n_samples,)
            Overall survival values used for coloring uncensored patients.
        comp_maps : np.ndarray, shape (n_samples, n_components), optional
            PCA component matrix. If None, then plot the components estimated
            from the training set.
        censorship : array-like, shape (n_samples,), optional
            Binary array indicating censoring status (1 = uncensored, 0 = censored).
            If None, all patients are assumed to be uncensored.
        save_dir : str, optional
            Directory to save the figure. If None, the figure is displayed instead.
        filename : str, optional
            Filename for saving the figure. Used only if `save_dir` is provided.
        cmap : str, default="viridis"
            Colormap for uncensored patients.
        figsize : tuple, default=(16, 12)
            Figure size in inches.
        dpi : int, default=100
            Resolution (dots per inch) for saving the figure.
        share_colorbar : bool, default=False
            If True, use a single colorbar for all plots. Otherwise, each subplot
            has its own colorbar.
        bbox_to_anchor : tuple, default=(.95, 0.05)
            Anchor position for placing the censored legend outside the figure.
        p : array_like of float
            For a better colorbar scaling due to outliers in survival times.
            Percentage or sequence of percentages for the percentiles to compute. 
            Values must be between 0 and 100 inclusive.
        clabel : str, optional, default="Overall Survival (OS)"
            Label of the colorbar to show.
        verbose : bool, optional, default=True
            If True, display information

        Notes
        -----
        - Uncensored patients are plotted with color intensity reflecting OS values.
        - Censored patients are plotted in light gray.
        - Reference lines are drawn through (0,0) in each subplot.
        - If `save_dir` is provided, the figure is saved to disk instead of shown.
        - Unused subplot axes (if grid > number of pairs) are hidden.

        Returns
        -------
        None
            Displays the figure or saves it to `save_dir/filename`.
        """
        if comp_maps is None:
            if not hasattr(self.TD_PCA, "training_scores"):
                raise RuntimeError("Components were not provided and were not not available in voxel space before projection.")
            comp_maps = self.TD_PCA.training_scores

        # Survival data checks
        overall_survival = np.array(overall_survival)
        if censorship is None:
            censorship = np.ones(overall_survival.shape)
        else:
            censorship = np.array(censorship)

        # Build all unique pairs
        pairs = list(itertools.combinations(components, 2))

        n_pairs = len(pairs)
        ncols = int(np.ceil(np.sqrt(n_pairs)))
        nrows = int(np.ceil(n_pairs / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes = np.atleast_1d(axes).ravel()  # flatten axes array

        cmap, vmin, vmax = Resample_ColorMap(cmap, overall_survival, p=p)
        scatters = []
        for ax, (i, j) in zip(axes, pairs):
            # Plot censored patients in light gray
            ax.scatter(
                comp_maps[censorship==0, i],
                comp_maps[censorship==0, j],
                edgecolor="black",
                facecolor="lightgray",
                alpha=0.8,
                s=10,
                label="Censored",
            )

            # Plot uncensored patients, colored by OS
            from matplotlib.colors import TwoSlopeNorm
            sc = ax.scatter(
                comp_maps[censorship==1, i],
                comp_maps[censorship==1, j],
                c=overall_survival[censorship==1],
                cmap=cmap,
                alpha=0.8,
                s=20,
                label="Uncensored",
                vmin=vmin,
                vmax=vmax,
            )
            scatters.append(sc)

            ax.set_xlabel(f"Component #{i+1}")
            ax.set_ylabel(f"Component #{j+1}")
            ax.spines[["top", "right"]].set_visible(False)

            if not share_colorbar:
                cbar = plt.colorbar(sc, ax=ax)
                cbar.set_label(clabel)

            ax.axhline(0, color="black", lw=0.8, ls="--")
            ax.axvline(0, color="black", lw=0.8, ls="--")

        # Add a single shared colorbar if requested
        if share_colorbar:
            fig.colorbar(
                scatters[0],
                ax=axes.tolist(),
                orientation="vertical",
                fraction=0.03,  # width of colorbar relative to figure
                pad=0.04,       # space between plot and colorbar
                label=clabel
            )
        
        censored_patch = mpatches.Patch(edgecolor="black", facecolor="lightgray", label="Censored", linewidth=1)
        fig.legend(
            handles=[censored_patch], 
            loc="lower right", 
            bbox_to_anchor=bbox_to_anchor, 
            frameon=False
        )

        # Hide unused axes if any
        for ax in axes[len(pairs):]:
            ax.axis("off")

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, filename)
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
            plt.close()
            if verbose:
                print(f"INFO:\tSaved surface plots to {save_path}")
        else:
            plt.show()

    def LowRank_convergence(self, maps, transformed_maps=None, max_components=None, eps=1e-12, n_jobs=1):
        """
        Evaluate convergence of low-rank PCA reconstruction.

        For each number of components from 1 up to `max_components`:
        - Reconstruct the data
        - Compute explained variance per subject
        - Compute correlation between original and reconstructed maps per subject

        Parallelization
        ---------------
        The loop over the number of components (`1 ... max_components`) is
        embarrassingly parallel: each reconstruction is independent. The argument
        `n_jobs` controls how many parallel workers are used:
        - n_jobs=1: run sequentially (default, safer for debugging).
        - n_jobs=-1: use all available cores.
        - n_jobs=k: use k cores.
        Be careful not to be overload the RAM capacities of your machine.

        Parameters
        ----------
        maps : np.ndarray, shape (n_samples, n_features)
            Original data in feature space (e.g., voxel intensities).
        transformed_maps : np.ndarray, shape (n_samples, n_components), optional
            Data already projected into PCA space. 
            If None, uses `self.TD_PCA.training_scores`.
        max_components : int, optional
            Maximum number of components to use for reconstruction. 
            If None, defaults to `self.n_components_`.
        eps : float, default=1e-12
            Small constant to avoid division by zero.
        n_jobs : int, default=1
            Number of parallel jobs for reconstruction across component counts.

        Returns
        -------
        explained_all : np.ndarray, shape (max_components, n_samples)
            Explained variance per subject for each rank.
        corrs_all : np.ndarray, shape (max_components, n_samples)
            Pearson correlation per subject for each rank.
        """
        if transformed_maps is None:
            transformed_maps = self.TD_PCA.training_scores  

        if max_components is None:
            max_components = self.n_components_

        n_subjects = transformed_maps.shape[0]

        def process_rank(n_c):
            low_rank = self.TD_PCA.reconstruct_LowRank(
                transformed_maps, 
                n_components=n_c, 
                mask_negative=True, 
                make_nan=False,
            )
            print(f"INFO:\tDone for {n_c} components.")

            """ # Residuals
            R = maps - low_rank
            
            # Sum of squares
            ss_res = np.sum(R ** 2, axis=1)
            ss_tot = np.sum((maps - np.mean(maps, axis=1)[:, None]) ** 2, axis=1) """
            
            # Explained variance
            explained = np.full(n_subjects, np.nan, dtype=float)
            """ mask_nonzero_var = ss_tot > eps
            explained[mask_nonzero_var] = (
                1 - (ss_res[mask_nonzero_var] / (ss_tot[mask_nonzero_var] + eps))
            )
             """
            # Correlation per subject
            corrs = np.full(n_subjects, np.nan, dtype=float)
            cosines = np.full(n_subjects, np.nan, dtype=float)
            for i in range(n_subjects):
                x = maps[i].ravel()
                y = low_rank[i].ravel()
                """ if np.std(x) > eps and np.std(y) > eps:
                    corrs[i] = np.corrcoef(x, y)[0, 1] """
                
                # Cosine similarity
                denom = (np.linalg.norm(x) * np.linalg.norm(y))
                if denom > eps:
                    cosines[i] = np.dot(x, y) / denom
            
            return explained, corrs, cosines

        # Run in parallel across component counts
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_rank)(n_c) for n_c in range(1, max_components + 1)
        )

        explained_all, corrs_all, cosines_all = zip(*results)
        explained_all = np.vstack(explained_all)
        corrs_all = np.vstack(corrs_all)
        cosines_all = np.vstack(cosines_all)

        return explained_all, corrs_all, cosines_all

        
