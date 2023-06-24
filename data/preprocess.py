
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

H5FILE_BIN = cfg['Preprocess']['FILE_PREFIX'] + 'bin.h5'
H5FILE_DT = cfg['Preprocess']['FILE_PREFIX'] + 'dt.h5'


def main():
    ''' datasets description:
    voxdims_000:    voxel dimensions/spacing_(case_index) in axes order
    gland_0000:     gland_(case index), in unit8 3d arrays
    targets_0000:   targets_(case index), in unit8 3d arrays
    dt_0000:        dt_(case index) for gland, float32 distance transform, inside(<0) and outside(>0)
    dt_0000_00:     dt_(case index)_(target_index) for targets, float32 distance transform, inside(<0) and outside(>0)
    filenames_all:  stores indexed case file names
    all written in a single h5 file, separating targets and gland (and binary and dt) for memory when reading
    '''
    fh5_bin = h5py.File(H5FILE_BIN, 'w')
    fh5_dt = h5py.File(H5FILE_DT, 'w')
    write_h5_bin = lambda fn, d: fh5_bin.create_dataset(fn,d.shape,dtype=d.dtype,data=d)
    write_h5_dt = lambda fn, d: fh5_dt.create_dataset(fn,d.shape,dtype=d.dtype,data=d)
    filenames_all = []
    for idx, filename in enumerate([os.path.join(LOCAL_PATH_TARGETS,f) for f in os.listdir(LOCAL_PATH_GLAND)]):  

        if not os.path.isfile(filename):
            print('WARNING: %s cannot be found or open.' % filename)
            continue
        #TODO: check the image exists
        case_idx = len(filenames_all)
        filenames_all += [filename.split('/')[-1]]

        targets = sitk.ReadImage(filename, outputPixelType=sitk.sitkUInt8)
        voxdims = targets.GetSpacing()[::-1]
        fh5_bin.create_dataset('/voxdims_%04d' % case_idx,len(voxdims),data=voxdims)

        two_way_dt = lambda x: (distance_transform(1-x,sampling=voxdims)-distance_transform(x,sampling=voxdims)).astype('float32')
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
            write_h5_dt('/dt_%04d_%02d' % (case_idx,idx+1), dt_target)

        write_h5_bin('/gland_%04d' % case_idx, gland_array)
        write_h5_bin('/targets_%04d' % case_idx, targets_array)
        write_h5_dt('/dt_%04d' % case_idx, dt_gland)

    # after for loop
    fh5_bin.create_dataset('filenames_all',len(filenames_all),data=filenames_all)
    fh5_bin.flush()
    fh5_bin.close()
    fh5_dt.flush()
    fh5_dt.close()
    print("%d data preprocessed and saved at %s, %s." % (case_idx+1,H5FILE_BIN, H5FILE_DT))


if __name__ == "__main__":
    main()