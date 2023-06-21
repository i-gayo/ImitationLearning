
import os

USER_SERVER = 'pt1'  #'128.16.4.203'
REMOTE_PATH = '"/raid/candi/Iani/MRes_project/Reinforcement\ Learning/DATASETS"'
LOCAL_PATH = 'data_tmp'

if not os.path.isdir(LOCAL_PATH):
    os.mkdir(LOCAL_PATH)

os.system('scp -r '+USER_SERVER+':'+REMOTE_PATH+' '+LOCAL_PATH)
