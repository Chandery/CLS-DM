import h5py
import numpy as np

file = "/disk/cdy/LIDC-HDF5-256/LIDC-IDRI-0335.20000101.30054.1/ct_xray_data.h5"

f = h5py.File(file, 'r')

img = f["ct"]
img = np.array(img)
img = np.expand_dims(img, 0)

from monai.transforms import SaveImage
import numpy as np
# import SimpleITK as sitk
import NibabelWriter as nib
print(img.shape)
# saver = SaveImage(
#         # output_dir=,
#         output_ext=".nii.gz",
#         separate_folder=False,
#         # output_dtype=np.uint8,
#         resample=False,
#         squeeze_end_dims=True,
#         writer="NibabelWriter",
#     )
# saver(img, filename="1")
nib.write_nifti(img, "test.nii.gz")