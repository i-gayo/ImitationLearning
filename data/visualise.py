
import os

import SimpleITK as sitk
import matplotlib.pyplot as plt


LOCAL_PATH_IMAGE = 'data_tmp/DATASETS/t2w'
LOCAL_PATH_GLAND = 'data_tmp/DATASETS/prostate_mask'
LOCAL_PATH_TARGETS = 'data_tmp/DATASETS/lesion'

IDX = 46


filenames = os.listdir(LOCAL_PATH_IMAGE)

image = sitk.ReadImage(os.path.join(LOCAL_PATH_IMAGE,filenames[IDX]))
gland = sitk.ReadImage(os.path.join(LOCAL_PATH_GLAND,filenames[IDX]))
targets = sitk.ReadImage(os.path.join(LOCAL_PATH_TARGETS,filenames[IDX]))

print('Pixel dimension: %s-%s-%s.' % image.GetSpacing())
print('Image size: %d-%d-%d.' % image.GetSize())

image = sitk.GetArrayFromImage(image)
image = (image-image.min())/(image.max()-image.min())*255
gland = sitk.GetArrayFromImage(gland)
targets = sitk.GetArrayFromImage(targets)


for idd in range(image.shape[0]):

    slice_targets = targets[idd,...]
    if sum(sum(slice_targets))==0: continue

    slice = image[idd,...] * (gland[idd,...]*0.5+0.5) * (1-slice_targets)
    sitk.WriteImage(sitk.GetImageFromArray(slice.astype('uint8')), 'test_%d.jpg'%idd)
