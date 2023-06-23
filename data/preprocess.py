
import os
import configparser

import SimpleITK as sitk






def main():

    cfg = configparser.ConfigParser()
    cfg.read('config.ini')
    LOCAL_PATH_IMAGE = cfg['Data']['LOCAL_PATH_IMAGE']
    #LOCAL_PATH_GLAND = cfg['Data']['LOCAL_PATH_GLAND']
    LOCAL_PATH_TARGETS = cfg['Data']['LOCAL_PATH_TARGETS']
    TARGET_VOLUME_LOW = eval(cfg['Preprocess']['TARGET_VOLUME_LOW'])

    for idx,filename in enumerate([os.path.join(LOCAL_PATH_TARGETS,f) for f in os.listdir(LOCAL_PATH_IMAGE)]):
        if not os.path.isfile(filename):
            print('WARNING: %s cannot be found or open.' % filename)
            continue

        targets = sitk.ReadImage(filename,outputPixelType=sitk.sitkUInt8)
        targets_components = sitk.Cast(sitk.ConnectedComponent(targets),sitk.sitkUInt8)
        targets_components = sitk.RelabelComponent(targets_components, sortByObjectSize=True)
        target_components_array = sitk.GetArrayFromImage(targets_components)
        num = target_components_array.max()
        volumes = [(target_components_array==(id+1)).sum() for id in range(num)]
        
        ## remove small islands
        for idx, v in enumerate(volumes):
            if v < TARGET_VOLUME_LOW:
                target_components_array[target_components_array==(num-v)] = 0

        ## compute DT (?)

    print("Preprocessing done")


if __name__ == "__main__":
    main()