


## 1. data 

Do the following independent of the main code, with the `data` as working directory

### 1.1 create and install data conda env
```bash
conda create -n imitation-data matplotlib scipy h5py && conda activate imitation-data && pip install SimpleITK
```

### 1.2 download raw data in `data_tmp` subfolder
```bash
python download.py
```
Optional: examine data statistics and plot example data
```bash
python stats.py
python visualise.py
```

### 1.3 configure using `config.ini` file

### 1.4 pre-process data and save in `pre_processed_*.h5` file
```bash
python preprocess.py
```


## 2. generate episodes

### 2.1 create and install runtime conda env
```bash
conda create -n imitation-run && conda activate imitation-run && conda install h5py pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
```
