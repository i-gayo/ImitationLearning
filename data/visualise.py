
import os

import nibabel as nib
import matplotlib.pyplot as plt


LOCAL_PATH_IMAGE = 'data_tmp/DATASETS/t2w'
LOCAL_PATH_GLAND = 'data_tmp/DATASETS/prostate_mask'
LOCAL_PATH_TARGETS = 'data_tmp/DATASETS/lesion'

IDX = 10


filenames = os.listdir(LOCAL_PATH_IMAGE)

image = nib.load(os.path.join(LOCAL_PATH_IMAGE,filenames[IDX]))
gland = nib.load(os.path.join(LOCAL_PATH_GLAND,filenames[IDX]))
targets = nib.load(os.path.join(LOCAL_PATH_TARGETS,filenames[IDX]))

pixdim = image.header['pixdim']
print('Pixel dimension: %s.' % pixdim[1:4])
print('Image size: %d-%d-%d.' % image.shape[:3])


idd = int(image.shape[2]/2)
slice = image.get_fdata()[:,:,idd] * (gland.get_fdata()[:,:,idd]*0.5+0.5)

plt.figure()
plt.imshow(slice, cmap='gray')
plt.axis('off')
#plt.show()
plt.savefig('test.jpg',bbox_inches='tight')
plt.close()
