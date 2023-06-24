


## 1. create and install conda env for data download, check and preprocessing
'''bash
conda create -n imitation-data matplotlib scipy h5py && conda activate imitation-data && pip install SimpleITK
'''

## 2. download raw data in `data_tmp`
```bash
python download.py
```

## 4. configure using `config.ini` file

## 3. pre-process data and save in `pre_processed_*.h5` file
'''bash
python preprocess.py
'''
