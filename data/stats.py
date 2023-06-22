
import os

import SimpleITK as sitk
import matplotlib.pyplot as plt


LOCAL_PATH_IMAGE = 'data_tmp/DATASETS/t2w'
LOCAL_PATH_GLAND = 'data_tmp/DATASETS/prostate_mask'
LOCAL_PATH_TARGETS = 'data_tmp/DATASETS/lesion'


FLAG_PLOT_SLICES = False 
if FLAG_PLOT_SLICES:
    PLOT_PATH = 'plots'
    if not os.path.isdir(PLOT_PATH):
        os.mkdir(PLOT_PATH)


## first, compute statistics for all lesions
volumes_all = []
num_all = []
for idx,filename in enumerate([os.path.join(LOCAL_PATH_TARGETS,f) for f in os.listdir(LOCAL_PATH_IMAGE)]):
    if not os.path.isfile(filename):
        print('WARNING: %s cannot be found or open.' % filename)
        continue

    targets = sitk.ReadImage(filename,outputPixelType=sitk.sitkUInt8)
    targets_components = sitk.Cast(sitk.ConnectedComponent(targets),sitk.sitkUInt8)
    target_components_array = sitk.GetArrayFromImage(targets_components)
    num = target_components_array.max()

    if FLAG_PLOT_SLICES:
        for idd in range(targets.GetSize()[2]): 
            slice = target_components_array[idd,...]
            if sum(sum(slice))==0: continue
            
            sitk.WriteImage(sitk.GetImageFromArray((slice/num*255).astype('uint8')), os.path.join(PLOT_PATH,'case%d_%d.jpg'%(idx,idd)))
    
    volumes = [(target_components_array==(id+1)).sum() for id in range(num)]
    volumes_all += volumes
    num_all += num

plt.figure(), plt.hist(volumes_all,bins=200), plt.savefig('hist_lesion.jpg'), plt.close()
