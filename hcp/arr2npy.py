"Export HCP dense connectome to a numppy array"
import nibabel as nib
import numpy as np

# Read connenctivity matrix
dcon_img = nib.load("../data/HCP_S1200_1003_rfMRI_MSMAll_groupPCA_d4500ROW_zcorr.dconn.nii")
dcon_mtx = dcon_img.get_fdata()
filename = "../data/HCP_S1200_1003_rfMRI_MSMAll_groupPCA_d4500ROW_zcorr.npy"

with open(filename, "wb") as f:
    np.save(f, dcon_mtx)
