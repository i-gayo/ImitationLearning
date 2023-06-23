
import os
import configparser

import SimpleITK as sitk


cfg = configparser.ConfigParser()
cfg.read('config.ini')
LOCAL_PATH_IMAGE = cfg['Data']['LOCAL_PATH_IMAGE']
LOCAL_PATH_GLAND = cfg['Data']['LOCAL_PATH_GLAND']
LOCAL_PATH_TARGETS = cfg['Data']['LOCAL_PATH_TARGETS']

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
