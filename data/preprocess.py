
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

H5FILE_BIN_GLAND = cfg['Preprocess']['FILE_PREFIX'] + 'bin_g.h5'
H5FILE_BIN_TARGETS = cfg['Preprocess']['FILE_PREFIX'] + 'bin_t.h5'
H5FILE_DT_GLAND = cfg['Preprocess']['FILE_PREFIX'] + 'dt_g.h5'
H5FILE_DT_TARGETS = cfg['Preprocess']['FILE_PREFIX'] + 'dt_t.h5'


def main():

    ''' datasets description:
    voxdims_000:    voxel dimensions/spacing_(case_index) in axes order
    gland_0000:     gland_(case index), in unit8 3d arrays
    targets_0000:   targets_(case index), in unit8 3d arrays
    dt_0000:        dt_(case index) for gland, float distance transform, inside(<0) and outside(>0)
    dt_0000_00:     dt_(case index)_(target_index) for targets, float distance transform, inside(<0) and outside(>0)
    filenames_all:  stores indexed case file names
    all written in a single h5 file, separating targets and gland (and binary and dt) for memory when reading
    '''

    fh5_bin_g = h5py.File(H5FILE_BIN_GLAND, 'w')
    fh5_bin_t = h5py.File(H5FILE_BIN_TARGETS, 'w')
    fh5_dt_g = h5py.File(H5FILE_DT_GLAND, 'w')
    fh5_dt_t = h5py.File(H5FILE_DT_TARGETS, 'w')
    write_h5_array = lambda fn, dn, d: fn.create_dataset(dn, d.shape,dtype=d.dtype, data=d)

    filenames_all = []
    for idx, filename in enumerate([os.path.join(LOCAL_PATH_TARGETS,f) for f in os.listdir(LOCAL_PATH_GLAND)]):  
        #DEBUG: if idx>10: break

        if not os.path.isfile(filename):
            print('WARNING: %s cannot be found or open.' % filename)
            continue
        #TODO: check the image exists
        case_idx = len(filenames_all)
        filenames_all += [filename.split('/')[-1]]

        ## read targets
        targets = sitk.ReadImage(filename, outputPixelType=sitk.sitkUInt8)
        voxdims = targets.GetSpacing()[::-1]
        fh5_bin_g.create_dataset('/voxdims_%04d' % case_idx,len(voxdims),data=voxdims)

        two_way_dt = lambda x: (distance_transform(1-x,sampling=voxdims)-distance_transform(x,sampling=voxdims)).astype('float16')
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
                # update num and volumes after removal
                num = targets_array.max()
                volumes = [(targets_array==(id+1)).sum() for id in range(num)]
        write_h5_array(fh5_bin_t, '/targets_%04d' % case_idx, targets_array)

        ## read gland
        gland = sitk.ReadImage(
            os.path.join(LOCAL_PATH_GLAND,filename.split('/')[-1]), 
            outputPixelType=sitk.sitkUInt8 )
        #TODO: check meta data consistency
        gland_array = sitk.GetArrayFromImage(gland)
        write_h5_array(fh5_bin_g, '/gland_%04d' % case_idx, gland_array)

        ## compute DT        
        dt_gland = two_way_dt(gland_array)
        write_h5_array(fh5_dt_g, '/dt_%04d' % case_idx, dt_gland)
        #sitk.WriteImage(sitk.GetImageFromArray((abs(dt_gland[40,...])/abs(dt_gland).max()*255).astype('uint8')),'test_d.jpg')

        for idx, v in enumerate(volumes):
            dt_target = two_way_dt(targets_array==(idx+1))
            write_h5_array(fh5_dt_t, '/dt_%04d_%02d' % (case_idx,idx+1), dt_target)


    # after for loop
    fh5_bin_g.create_dataset('filenames_all',len(filenames_all),data=filenames_all)
    fh5_bin_g.close()
    fh5_bin_t.close()
    fh5_dt_g.close()
    fh5_dt_t.close()
    print("%d data preprocessed and saved at: \n%s \n%s \n%s \n%s." % (
        case_idx+1, 
        H5FILE_BIN_GLAND, H5FILE_BIN_TARGETS, 
        H5FILE_DT_GLAND, H5FILE_DT_TARGETS))


if __name__ == "__main__":
    main()
