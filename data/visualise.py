
import os

import SimpleITK as sitk
import matplotlib.pyplot as plt


LOCAL_PATH_IMAGE = 'data_tmp/DATASETS/t2w'
LOCAL_PATH_GLAND = 'data_tmp/DATASETS/prostate_mask'
LOCAL_PATH_TARGETS = 'data_tmp/DATASETS/lesion'

IDX = 10


filenames = os.listdir(LOCAL_PATH_IMAGE)

image = sitk.ReadImage(os.path.join(LOCAL_PATH_IMAGE,filenames[IDX]))
gland = sitk.ReadImage(os.path.join(LOCAL_PATH_GLAND,filenames[IDX]))
targets = sitk.ReadImage(os.path.join(LOCAL_PATH_TARGETS,filenames[IDX]))

print('Pixel dimension: %s-%s-%s.' % image.GetSpacing())
print('Image size: %d-%d-%d.' % image.GetSize())


idd = int(image.GetSize()[2]/2)
slice = sitk.GetArrayFromImage(image)[idd,...] * (sitk.GetArrayFromImage(gland)[idd,...]*0.5+0.5) * (1-sitk.GetArrayFromImage(targets)[idd,...])

plt.figure()
plt.imshow(slice, cmap='gray')
plt.axis('off')
#plt.show()
plt.savefig('test.jpg',bbox_inches='tight')
plt.close()
