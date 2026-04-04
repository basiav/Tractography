import os
import pandas as pd
import numpy as np
from tqdm import tqdm   
from typing import Optional, Union

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import AxesGrid

from scipy.stats import zscore

from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.compare import compare_survival
from sksurv.linear_model import CoxPHSurvivalAnalysis

import nibabel as nib

def rewrite_subjectID(subject_ID):
    subject_3digits = subject_ID.split("-")
    subject = "-".join(subject_3digits[:-1])
    digits = str(subject_3digits[-1])
    return subject + "-0" + digits

def process_UCSF(root, stream_th, UCSF_dir=None):
    if UCSF_dir is None:
        UCSF_dir = "Glioblastoma_UCSF-PDGM_v3-20230111"

    root_ucsf = os.path.join(root, UCSF_dir)
    TDstats_ucsf = pd.read_csv(os.path.join(root_ucsf, f"TDMaps_Grade-IV/demographics-TDMaps_streamTH-{stream_th}.csv"))
    morphology_ucsf = pd.read_csv(os.path.join(root_ucsf, f"TDMaps_Grade-IV/morphology-tissues.csv"))

    TDstats_ucsf = TDstats_ucsf.loc[TDstats_ucsf["Final pathologic diagnosis (WHO 2021)"]=="Glioblastoma  IDH-wildtype"] 
    TDstats_ucsf = TDstats_ucsf.loc[TDstats_ucsf["OS"].fillna('unknown')!='unknown']
    morphology_ucsf = morphology_ucsf.loc[morphology_ucsf["Final pathologic diagnosis (WHO 2021)"]=="Glioblastoma  IDH-wildtype"] 
    morphology_ucsf = morphology_ucsf.loc[morphology_ucsf["OS"].fillna('unknown')!='unknown']

    common_cols = ['ID', 'Sex', 'Age at MRI', 'WHO CNS Grade', 
        'Final pathologic diagnosis (WHO 2021)', 'MGMT status', 'MGMT index',
        '1p/19q', 'IDH', '1-dead 0-alive', 'OS', 'EOR',
        'Biopsy prior to imaging', 'BraTS21 ID', 'BraTS21 Segmentation Cohort',
        'BraTS21 MGMT Cohort', '# Labels'
    ]
    ucsf = pd.merge(TDstats_ucsf, morphology_ucsf, on=common_cols)

    print("UCSF cohort description --> N =", ucsf["1-dead 0-alive"].value_counts().sum())
    censored = (ucsf["1-dead 0-alive"]==0).sum()
    all_ucsf = ucsf["1-dead 0-alive"].value_counts().sum()
    print(f"Precentage of censoring: {round(100*censored/all_ucsf,2)}%\n")

    return ucsf, censored, all_ucsf

def process_UPENN(root, stream_th, UPENN_dir=None):
    if UPENN_dir is None:
        UPENN_dir = "Glioblastoma_UPENN-GBM_v2-20221024"

    root_upenn = os.path.join(root, UPENN_dir)
    TDstats_upenn = pd.read_csv(os.path.join(root_upenn, f"TDMaps_IDH1-WT/demographics-TDMaps_streamTH-{stream_th}.csv"))
    # TDstats_upenn = pd.read_csv("/home/joan/Desktop/PROJECTS/Glioblastomas/RESULTS-GBM_4-cohorts_UMAP-Risks/data-clinical_parcellation-hcpex_4-cohorts.csv")
    morphology_upenn = pd.read_csv(os.path.join(root_upenn, f"TDMaps_IDH1-WT/morphology-tissues.csv"))
        
    common_cols = ["ID", "Gender", "Age_at_scan_years", "Survival_from_surgery_days_UPDATED", "Survival_Status", "MGMT", "KPS", "GTR_over90percent"]
    upenn = pd.merge(TDstats_upenn, morphology_upenn, on=common_cols)

    print("UPENN cohort description --> N =", upenn["Survival_Status"].value_counts().sum())
    censored = (upenn["Survival_Status"]==0).sum()
    all_upenn = upenn["Survival_Status"].value_counts().sum()
    print(f"Precentage of censoring: {round(100*censored/all_upenn,2)}%\n")

    return upenn, censored, all_upenn

def process_data(root, stream_th, voxel_size, UCSF_dir=None, UPENN_dir=None, RHUH_dir=None, TCGA_dir=None, z_transform=0, cohort_ids=[0,1,2,3]):
    # ucsf, _, _ = process_UCSF(root, stream_th, UCSF_dir)
    # upenn, _, _ = process_UPENN(root, stream_th, UPENN_dir)

    # df = pd.read_csv("/home/joan/Desktop/PROJECTS/Glioblastomas/RESULTS-GBM_4-cohorts_UMAP-Risks/data-clinical_parcellation-hcpex_4-cohorts.csv")
    df = pd.read_csv("/home/joan/Desktop/PROJECTS/Barbara/Tract-Density_Components-Survival/data-clinical_parcellation-hcpex_4-cohorts.csv")
    site_groups = {site: data for site, data in df.groupby(df['ID'].str.split('-').str[0])}
    ucsf = site_groups.get('UCSF')
    upenn = site_groups.get('UPENN')
    rhuh = site_groups.get('RHUH')
    tcga = site_groups.get('TCGA')

    # UCSF
    df_ucsf = pd.DataFrame({
        "ID": ucsf["ID"].values,
        "sex": ucsf["sex"].values,
        "age": ucsf["age"].values,
        "site": np.zeros(len(ucsf), dtype=int) + cohort_ids[0],  # UCSF = 0
        "volume": ucsf["Whole tumor size (voxels)"].values * voxel_size,
        "ltdi": ucsf["Whole lesion TDMap"].values,
        "tdi": ucsf["Whole TDMap"].values,
        "OS": ucsf["OS (days) - corrected"].values,
        # "status": ucsf["1-dead 0-alive"],
        "status": ucsf["status"],
        # Added
        "EOR": ucsf["eor"],
        # "MGMT status": ucsf["MGMT status"]
        "MGMT": ucsf["mgmt"]
    })

    # UPENN
    df_upenn = pd.DataFrame({
        "ID": upenn["ID"].values,
        # "sex": upenn["Gender"].values,
        "sex": upenn["sex"],
        "age": upenn["age"].values,
        "site": np.zeros(len(upenn), dtype=int) + cohort_ids[1],  # UPENN = 1
        "volume": upenn["Whole tumor size (voxels)"].values * voxel_size,
        "ltdi": upenn["Whole lesion TDMap"].values,
        "tdi": upenn["Whole TDMap"].values,
        "OS": upenn["OS (days) - corrected"],
        # "status": upenn["Survival_Status"],
        # Added
        # "EOR": upenn["EOR"],
        # "MGMT status": upenn["MGMT"]
                # "status": ucsf["1-dead 0-alive"],
        "status": upenn["status"],
        # Added
        "EOR": upenn["eor"],
        # "MGMT status": ucsf["MGMT status"]
        "MGMT": upenn["mgmt"]
    })

    # DATA = pd.concat([df_ucsf, df_upenn], axis=0, ignore_index=True)
    df_rhuh = pd.DataFrame({
        "ID": rhuh["ID"].values,
        # "sex": rhuh["Gender"].values,
        "sex": rhuh["sex"],
        "age": rhuh["age"].values,
        "site": np.zeros(len(rhuh), dtype=int) + cohort_ids[2],  # RHUH = 2
        "volume": rhuh["Whole tumor size (voxels)"].values * voxel_size,
        "ltdi": rhuh["Whole lesion TDMap"].values,
        "tdi": rhuh["Whole TDMap"].values,
        "OS": rhuh["OS (days) - corrected"],
        # "status": rhuh["Survival_Status"],
        # Added
        # "EOR": rhuh["EOR"],
        # "MGMT status": rhuh["MGMT"]
                # "status": ucsf["1-dead 0-alive"],
        "status": rhuh["status"],
        # Added
        "EOR": rhuh["eor"],
        # "MGMT status": ucsf["MGMT status"]
        "MGMT": rhuh["mgmt"]
    })

    df_tcga = pd.DataFrame({
        "ID": tcga["ID"].values,
        # "sex": tcga["Gender"].values,
        "sex": tcga["sex"],
        "age": tcga["age"].values,
        "site": np.zeros(len(tcga), dtype=int) + cohort_ids[3],  # TCGA = 3
        "volume": tcga["Whole tumor size (voxels)"].values * voxel_size,
        "ltdi": tcga["Whole lesion TDMap"].values,
        "tdi": tcga["Whole TDMap"].values,
        "OS": tcga["OS (days) - corrected"],
        # "status": tcga["Survival_Status"],
        # Added
        # "EOR": tcga["EOR"],
        # "MGMT status": tcga["MGMT"]
                # "status": ucsf["1-dead 0-alive"],
        "status": tcga["status"],
        # Added
        "EOR": tcga["eor"],
        # "MGMT status": ucsf["MGMT status"]
        "MGMT": tcga["mgmt"]
    })

    DATA = pd.concat([df_ucsf, df_upenn, df_rhuh, df_tcga], axis=0, ignore_index=True)

    if z_transform == 1:
        for col in ["age", "volume", "ltdi", "tdi"]:
            DATA[col] = zscore(DATA[col])

    return DATA

def correct_OS(DATA, cohort_ids, n_perms=1000, alpha=0.05):
    # TODO: Eventually add plots and Scaled residuals and GT test

    ## Prepare survival data acording to scikit-survival format
    OS_STATS = []
    GROUP_STATS = []
    for i, cohort in enumerate(cohort_ids):

        diag = DATA[DATA["site"]==cohort]
        mask = ~np.isnan(diag["status"]) & ~np.isnan(diag["OS"])
        diag = diag[mask]

        OS_STATS.extend([(st, ovs) for st, ovs in zip(diag["status"]==1, diag["OS"].values)])
        GROUP_STATS.extend([i+1 for ovs in diag["OS"].values])
    OS_STATS = np.array(OS_STATS, dtype=[('event', 'bool'),('time', 'float')])
    chisquared, p_val, stats, covariance = compare_survival(OS_STATS, GROUP_STATS, return_stats=True)
    print(f"Log-rank test: chi-squared statistic of {chisquared} (p={p_val})")

    if p_val<=0.05:
        print("INFO: The effect of site was found to be significant, applying a correction procedure")
        ## Fit Cox model to estimate the effect of site
        Cmodel = CoxPHSurvivalAnalysis(n_iter=200)
        Cmodel.fit(DATA["site"].values.reshape(-1,1), OS_STATS)
        c_index = Cmodel.score(DATA["site"].values.reshape(-1,1), OS_STATS)
        pop = []
        for _ in tqdm(range(n_perms), desc="Performing permutation test for the C-index"):#
            perm_OS_STATS = np.random.permutation(OS_STATS)
            p_Cmodel = CoxPHSurvivalAnalysis()
            p_Cmodel.fit(DATA["site"].values.reshape(-1, 1), perm_OS_STATS)
            pop.append(p_Cmodel.score(DATA["site"].values.reshape(-1, 1), perm_OS_STATS))
        p_value = np.mean(np.array(pop) >= c_index)
        print(f"\tC-index of  {c_index} (p={p_value})")
        print(f"\tLog HR of the site effect = {Cmodel.coef_[0]}")

        ## Apply the correction to the survival data
        corrected_os = [] # Make sure that the order of the cohorts is robust
        for i, cohort in enumerate(cohort_ids):
            diag = DATA[DATA["site"]==cohort]
            mask = ~np.isnan(diag["status"]) & ~np.isnan(diag["OS"])
            diag = diag[mask]
            corrected_os.extend(diag["OS"] * np.exp(Cmodel.coef_[0]*cohort))
        DATA["OS (corrected)"] = corrected_os

        ## Perform control tests
        OS_STATS = []
        GROUP_STATS = []
        for i, cohort in enumerate(cohort_ids):

            diag = DATA[DATA["site"]==cohort]
            mask = ~np.isnan(diag["status"]) & ~np.isnan(diag["OS (corrected)"])
            diag = diag[mask]

            OS_STATS.extend([(st, ovs) for st, ovs in zip(diag["status"]==1, diag["OS (corrected)"].values)])
            GROUP_STATS.extend([i+1 for ovs in diag["OS (corrected)"].values])
        OS_STATS = np.array(OS_STATS, dtype=[('event', 'bool'),('time', 'float')])
        chisquared, p_val, stats, covariance = compare_survival(OS_STATS, GROUP_STATS, return_stats=True)
        print(f"Log-rank test after correction: chi-squared statistic of {chisquared} (p={p_val})\n")
    else:
        print("INFO: The effect of site was NOT found to be significant, copying OS data as it was")
        DATA["OS (corrected)"] = DATA["OS"]

    return DATA
# TDMaps_IDH1-WT
def load_maps(list_of_subjects, nFeatures, template, root, data_dir, tissue, tdmap, voxel_res, nthreads):
    maps = np.zeros((len(list_of_subjects), nFeatures))
    for i, subject in enumerate(list_of_subjects):
        file = f"{root}/{data_dir}/{subject}/maps/{subject}_tissue-{tissue}_{tdmap}_voxel-{voxel_res}.nii.gz"
        if not os.path.exists(file):
            original = f"{root}/{data_dir}/{subject}/maps/{subject}_tissue-{tissue}_{tdmap}.nii.gz"
            os.system(f"mrgrid {original} regrid -template {template} -voxel {voxel_res} {file} -q -nthreads {nthreads}")
        maps[i,:] = nib.load(file).get_fdata().flatten()
    return maps


def Resample_ColorMap(cmap, data, name='resampled_cmp', p=90):
    """
    Resample a colormap so that the data median corresponds to the center color (0.5).
    
    Parameters
    ----------
    cmap : matplotlib.colors.Colormap
        The original colormap to be resampled.
    data : array-like
        Data values used to compute the median and rescaling.
    name : str, default='resampled_cmp'
        Name for the new colormap.
    p : array_like of float
        Percentage or sequence of percentages for the percentiles to compute. 
        Values must be between 0 and 100 inclusive.
    
    Returns
    -------
    newcmap : matplotlib.colors.LinearSegmentedColormap
        The resampled colormap centered at the data median.
    
    Notes
    -----
    Adapted from: https://stackoverflow.com/questions/7404116/
    """
    data = np.asarray(data)
    vmin, vmax = np.min(data), np.max(data)
    percentile = np.percentile(data, p)

    # Normalize median into [0,1]
    if vmax == vmin:
        mid = 0.5
    else:
        mid = (percentile - vmin) / (vmax - vmin)

    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    # Regular and shifted indices
    reg_index = np.linspace(0, 1, 257)
    shift_index = np.hstack([
        np.linspace(0.0, mid, 128, endpoint=False), 
        np.linspace(mid, 1.0, 129, endpoint=True)
    ])

    # Build color dictionary
    cdict = {'red': [], 'green': [], 'blue': [], 'alpha': []}
    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)
        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    return newcmap, vmin, vmax

def unflatten_maps(
    TD_maps: Union[np.ndarray, list],
    template: Union[str, nib.Nifti1Image],
    mask_flat: Optional[np.ndarray] = None,
    make_nan: bool = False,
    return_nifti: bool = False
):
    """
    Unflatten flattened voxel maps (n_samples x n_features) to a 4D volume
    that matches the provided NIfTI template shape.

    Parameters
    ----------
    TD_maps : array-like, shape (n_samples, n_features) or (n_features,)
        Flattened maps. If 1D, treated as a single sample.
    template : str or nibabel.Nifti1Image
        NIfTI template (or path to .nii/.nii.gz). Its shape defines the 3D grid.
    mask_flat : array-like of bool or int, shape (n_voxels,), optional
        Boolean mask (flattened) indicating which voxels in the template were kept.
        If provided, n_features must equal mask_flat.sum(). If None, it is assumed
        TD_maps already cover every voxel of the template (n_features==prod(shape)).
    make_nan : bool, default=False
        If True, voxels outside the mask (when mask_flat provided) are filled with NaN;
        otherwise they are filled with 0.
    return_nifti : bool, default=False
        If True, returns a nibabel.Nifti1Image (with data shape (X, Y, Z, n_samples)).
        Otherwise returns a numpy array of shape (X, Y, Z, n_samples).

    Returns
    -------
    comp_4d_or_img : np.ndarray or nibabel.Nifti1Image
        4D volume with spatial dims first and sample index on the 4th axis.
        Shape = (X, Y, Z, n_samples) if numpy array; if `return_nifti` the Nifti image
        uses the affine/header from `template`.

    Raises
    ------
    ValueError
        If input shapes are incompatible (e.g., wrong number of features).
    """

    # Resolve template (accept path or nib image)
    if isinstance(template, str):
        if not os.path.exists(template):
            raise FileNotFoundError(f"Template file not found: {template}")
        tpl_img = nib.load(template)
    elif isinstance(template, nib.Nifti1Image) or hasattr(template, "get_fdata"):
        tpl_img = template
    else:
        raise TypeError("`template` must be a nibabel image or a path to a .nii(.gz) file.")

    tpl_shape = tpl_img.get_fdata().shape  # (X, Y, Z)
    n_voxels = int(np.prod(tpl_shape))

    # Convert maps to numpy and ensure 2D
    X = np.asarray(TD_maps)
    if X.ndim == 1:
        X = X[np.newaxis, :]
    if X.ndim != 2:
        raise ValueError("TD_maps must be 1D (single sample) or 2D (n_samples x n_features).")

    n_samples, n_features = X.shape

    # If mask provided, validate and expand to full voxels
    if mask_flat is not None:
        mask = np.asarray(mask_flat, dtype=bool).ravel()
        if mask.size != n_voxels:
            raise ValueError(f"mask_flat length ({mask.size}) does not match template voxels ({n_voxels}).")
        n_masked = mask.sum()
        if n_features != n_masked:
            raise ValueError(f"TD_maps has {n_features} features but mask contains {n_masked} True voxels.")
        # Create full flattened space and fill masked columns
        fill_value = np.nan if make_nan else 0.0
        full_flat = np.full((n_samples, n_voxels), fill_value, dtype=X.dtype)
        full_flat[:, mask] = X
    else:
        # No mask: expect full coverage
        if n_features != n_voxels:
            raise ValueError(
                f"TD_maps has {n_features} features but template has {n_voxels} voxels. "
                "Provide mask_flat if TD_maps are masked."
            )
        full_flat = X

    # Reshape to (n_samples, X, Y, Z) then transpose to (X, Y, Z, n_samples)
    full_4d_samples_first = full_flat.reshape((n_samples,) + tpl_shape)
    comp_4d = np.transpose(full_4d_samples_first, (1, 2, 3, 0))  # (X, Y, Z, n_samples)

    if return_nifti:
        comp_img = nib.Nifti1Image(comp_4d, affine=tpl_img.affine, header=tpl_img.header)
        return comp_img

    return comp_4d
