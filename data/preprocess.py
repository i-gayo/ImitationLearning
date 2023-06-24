
import os
import configparser

import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt as distance_transform 
import h5py


cfg = configparser.ConfigParser()
cfg.read('config.ini')
LOCAL_PATH_IMAGE = cfg['Data']['LOCAL_PATH_IMAGE']
LOCAL_PATH_GLAND = cfg['Data']['LOCAL_PATH_GLAND']
LOCAL_PATH_TARGETS = cfg['Data']['LOCAL_PATH_TARGETS']
TARGET_VOLUME_LOW = eval(cfg['Preprocess']['TARGET_VOLUME_LOW'])

H5FILE = cfg['Preprocess']['FILE_PREFIX'] + '.h5'


def main():
    '''
    voxdims are voxel dimensions/spacing in axes order
    gland_0000 are gland(case index), in unit8 3d arrays
    targets_0000 are targets_(case index), in unit8 3d arrays
    dt_0000 are dt_(case index) for gland, float32 distance transform, inside(<0) and outside(>0)
    dt_0000_00 are dt_(case index)_(target_index) for targets, float32 distance transform, inside(<0) and outside(>0)
    filenames store indexed case file names
    all written in a single h5 file, separating targets and gland (and binary and dt) for memory when reading
    '''
    fh5 = h5py.File(H5FILE, 'w')
    write_h5_data = lambda fn, d: fh5.create_dataset(fn,d.shape,dtype=d.dtype,data=d)
    filenames_all = []
    for idx, filename in enumerate([os.path.join(LOCAL_PATH_TARGETS,f) for f in os.listdir(LOCAL_PATH_GLAND)]):  

        if not os.path.isfile(filename):
            print('WARNING: %s cannot be found or open.' % filename)
            continue
        #TODO: check the image exists

        targets = sitk.ReadImage(filename, outputPixelType=sitk.sitkUInt8)
        voxdims = targets.GetSpacing()[::-1]
        two_way_dt = lambda x: distance_transform(1-x,sampling=voxdims)-distance_transform(x,sampling=voxdims)
        targets_array = sitk.GetArrayFromImage(
            sitk.RelabelComponent(
            sitk.Cast(sitk.ConnectedComponent(targets),sitk.sitkUInt8), 
            sortByObjectSize=True))
        num = targets_array.max()
        volumes = [(targets_array==(id+1)).sum() for id in range(num)]
        
        ## remove small islands
        for idx, v in enumerate(volumes):
            if v < TARGET_VOLUME_LOW:
                targets_array[targets_array==(idx+1)] = 0
                num = targets_array.max()
                volumes = [(targets_array==(id+1)).sum() for id in range(num)]

        ## read gland
        gland = sitk.ReadImage(
            os.path.join(LOCAL_PATH_GLAND,filename.split('/')[-1]), 
            outputPixelType=sitk.sitkUInt8 )
        #TODO: check meta data consistency
        gland_array = sitk.GetArrayFromImage(gland)

        ## compute DT        
        dt_gland = two_way_dt(gland_array)

        for idx, v in enumerate(volumes):
            dt_target = two_way_dt(targets_array==(idx+1))
            #sitk.WriteImage(sitk.GetImageFromArray((d[40,...]/d.max()*255).astype('uint8')),'test_d.jpg')
            write_h5_data('/dt_%04d_%02d' % (case_idx,idx+1), dt_target)

        write_h5_data('/gland_%04d' % case_idx, gland_array)
        write_h5_data('/targets_%04d' % case_idx, targets_array)
        write_h5_data('/dt_%04d' % case_idx, dt_gland)

        filenames_all += filename.split('/')[-1]
        case_idx = len(filenames_all)



    print("Preprocessing done")


if __name__ == "__main__":
    main()