import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm
import json
from joblib import Parallel, delayed

import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.cm as cm
import matplotlib.colors as mcolors

import seaborn as sns

import nibabel as nib

import nilearn.plotting as plotting

from scipy.stats import pearsonr, ttest_ind, mannwhitneyu, zscore, sem, f_oneway, permutation_test

import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import fdrcorrection
from statsmodels.duration.hazard_regression import PHReg

from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.compare import compare_survival
from sksurv.linear_model import CoxPHSurvivalAnalysis

from lifelines.statistics import proportional_hazard_test
from lifelines import CoxPHFitter
from lifelines.utils.lowess import lowess

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

import statsmodels.formula.api as smf

## Relative imports
from utils.utils import rewrite_subjectID, process_data, correct_OS, load_maps


## Directories and paths
voxel_res = 1.5
root =  "/home/joan/Desktop/PROJECTS/Glioblastomas" # "D:\JoanFR_Sano"
ucsf_dir = "Glioblastoma_UCSF-PDGM_v3-20230111"
upenn_dir = "Glioblastoma_UPENN-GBM_v2-20221024"
results_root = os.path.join(root,f"{ucsf_dir}/Figures/TDMaps_Grade-IV/PCA")
os.makedirs(results_root, exist_ok=True)
MNI_DIR = "/home/joan/Documents/MNI_ICBM_2009b_NLIN_ASYM" # "C:/Users/user/Documents/Data/MNI"
atlas_lobes = f"/home/joan/Documents/Parcellation/USCLobesMNI-ICBM-2009b-NLIN-ASYM/BCI-DNI_brain.labelxMNI-ICBM-2009b-NLIN-ASYM__Warped.nii.gz"
lobe_labels = f"/home/joan/Documents/Parcellation/USCLobesMNI-ICBM-2009b-NLIN-ASYM/BCI-DNI_brain.labels.txt"
atlas_wm = f"/home/joan/Documents/Parcellation/XTRACT_MNI-ICBM-2009b-NLIN-ASYM/xtract-tract-atlases-maxprob5-1mmxMNI-ICBM-2009b-NLIN-ASYM__Warped.nii.gz"
WM_labels = f"/home/joan/Documents/Parcellation/XTRACT_MNI-ICBM-2009b-NLIN-ASYM/XTRACT.labels.txt"
T1_file = os.path.join(MNI_DIR, f"T1_{voxel_res}mm.nii")
T1 = nib.load(T1_file)
T1_brain = nib.load(os.path.join(MNI_DIR, f"T1_{voxel_res}mm_brain_mask.nii"))

Ni, Nj, Nk = T1.get_fdata().shape
tissue = "whole"
tdmap = "TDMap-lesion"

## Other
fmt = "svg"
dpi = 300
stream_th = 0
daysXmonth = 365/12
voxel_size = (0.5**3) * (1/1000) # 0.5 (mm³/voxel) X 0.001 (cm³/mm³)   
sites = [0, 1]
n_jobs = 16
nthreads = 2

## Load and process clinical data
DATA = correct_OS(
    process_data(
        root,
        stream_th,
        voxel_size,
        UCSF_dir=ucsf_dir,
        UPENN_dir=upenn_dir,
        z_transform=0,
        cohort_ids=sites
    ),
    cohort_ids=sites,
    n_perms=10
)

# Process and load UCSF maps
ucsf_ids = [rewrite_subjectID(id) for id in DATA.loc[DATA["site"]==0]["ID"].values]
results = Parallel(n_jobs=n_jobs)(
    delayed(load_maps)(
        list_of_subjects=[subj_id],
        nFeatures=Ni * Nj * Nk,
        template=T1_file,
        root=root,
        data_dir=f"{ucsf_dir}/TDMaps_Grade-IV",
        tissue=tissue,
        tdmap=tdmap,
        voxel_res=voxel_res,
        nthreads=nthreads
    ) for subj_id in tqdm(ucsf_ids, desc="Loading UCSF maps")
)
maps_ucsf = np.vstack(results) # Nsubjects x nFeatures
maps_ucsf = np.where(maps_ucsf>=0, maps_ucsf, 0) # The resampling has introduced some small amount of negative numbers ## Be careful and double check
 
# Process and load UPENN maps
upenn_ids = DATA.loc[DATA["site"]==1]["ID"].values
results = Parallel(n_jobs=n_jobs)(
    delayed(load_maps)(
        list_of_subjects=[subj_id],
        nFeatures=Ni * Nj * Nk,
        template=T1_file,
        root=root,
        data_dir=f"{upenn_dir}/TDMaps_IDH1-WT",
        tissue=tissue,
        tdmap=tdmap,
        voxel_res=voxel_res,
        nthreads=nthreads
    ) for subj_id in tqdm(upenn_ids, desc="Loading UPENN maps")
)
maps_upenn = np.vstack(results) # Nsubjects x nFeatures
maps_upenn = np.where(maps_upenn>=0, maps_upenn, 0) # The resampling has introduced some small amount of negative numbers ## Be careful and double check

maps_all = np.vstack((maps_ucsf, maps_upenn))

# Load CORRECTED survival data
os_all = DATA["OS (corrected)"] / daysXmonth
status_all = DATA["status"] 

os_ucsf = DATA.loc[DATA["site"]==0]["OS (corrected)"] / daysXmonth
status_ucsf = DATA.loc[DATA["site"]==0]["status"] 

os_upenn = DATA.loc[DATA["site"]==1]["OS (corrected)"] / daysXmonth
status_upenn = DATA.loc[DATA["site"]==1]["status"] 

from TractDensityComponents import TD_PCA
from components import Projections

# Normalization strategy
for masking_strategy in [True]:
    print("MASKING STRATEGY: ", masking_strategy)
    for normalization in [None, 'z_norm', 'min_max']:
    #normalization = 'z_norm'

        # Directories to save the results and outputs
        save_dir = os.path.join(results_root, f"PCA-TD_Norm-{normalization}_Train-UCSF_Mask-{masking_strategy}")
        os.makedirs(save_dir, exist_ok=True)

        # Create the PCA object 
        TD_pca = TD_PCA(
            training_maps=maps_ucsf, # UCSF as training
            template=T1,             # 1.5 mm template on MNI
            # 1.5 mm brain mask on MNI to speed up the calculations
            binary_mask=T1_brain if masking_strategy else None,   
            n_components=120         # Number of components to fit
        )
        # Fit the PCA on the training data
        cmps_ucsf = TD_pca.fit_pca_maps(
            norm=normalization       # Determine the normalization strategy
        )

        # Obtain the explained variance
        thresholds = TD_pca.explained_variance_plot(
            show=False,
            save_dir=save_dir, 
            filename=f'Explained-variance_Cohort-UCSF.{fmt}',
            dpi=300
        )

        # Transform the UPENN maps (i.e., test) under the same conditions
        cmps_upenn = TD_pca.transform_maps(
            maps=maps_upenn         
        )

        # Create an object to project, analyze, and reconstruct the maps
        Project = Projections(
            TD_pca
        )

        # Plot PCA loadings for easier inspections
        save_dir_loadings = os.path.join(save_dir,"PCA-Loadings")
        os.makedirs(save_dir_loadings, exist_ok=True)
        print("INFO:\tProjecting PCA Loadings to volumes and surfaces")

        # Save PCa loadings to NIFTI file
        Project.components_to_brain(
            save_dir=save_dir, 
            filename="PCA-Loadings_Cohort-UCSF.nii.gz",
            verbose=False
        )

        N_comps_to_plots = 15
        for cp in range(N_comps_to_plots):
            # Volumetric plots
            Project.plot_component_volumes(
                background_image=T1,
                component_idx=cp, 
                threshold=None, 
                save_dir=save_dir_loadings, 
                filename=f"PCA-Loadings_Component-{cp+1}_Volume-maps.{fmt}", 
                cmap="RdBu",
                dpi=300,
                colorbar=False,
                verbose=False
            )
            # Pial surface projections
            Project.plot_component_surface(
                component_idx=cp, 
                threshold=None,
                save_dir=save_dir_loadings, 
                filename=f"PCA-Loadings_Component-{cp+1}_Surface-projections.{fmt}", 
                cmap="RdBu",
                alpha=1,
                dpi=300,
                colorbar=False,
                verbose=False
            )

        component_pairs = [0,1,2,3,4,5]
        print("INFO:\tInspecting 2D projections of the PCA Scores")
        # PCA scores for the UCSF cohort (i.e., train)
        Project.inspect_scores(
            components=component_pairs, 
            overall_survival=os_ucsf, 
            censorship=status_ucsf,
            cmap="RdYlBu", 
            figsize=(20, 16), 
            share_colorbar=True, 
            bbox_to_anchor=(.95, 0.05),
            save_dir=save_dir, 
            filename=f"PCA-Scores_Cohort-UCSF_Mode-train.{fmt}", 
            dpi = 300,
            p=50, # Percentile of OS to centre de colorbar
            clabel="Overall survival (OS; months)",
            verbose=False
        )

        # 2D Projections of the PCA scores for the UPENN cohort (i.e., test)
        Project.inspect_scores(
            components=component_pairs, 
            overall_survival=os_upenn, 
            comp_maps=cmps_upenn, 
            censorship=status_upenn,
            cmap="RdYlBu", 
            figsize=(20, 16), 
            share_colorbar=True, 
            bbox_to_anchor=(.95, 0.05),
            save_dir=save_dir, 
            filename=f"PCA-Scores_Cohort-UPENN_Mode-test.{fmt}",
            dpi=300,
            p=50,  # Percentile of OS to centre de colorbar
            clabel="Overall survival (OS; months)",
            verbose=False
        )
    print("\n\n")