
import os

user_server = 'pt1'  #'128.16.4.203'
remote_path = '"/raid/candi/Iani/MRes_project/Reinforcement\ Learning/DATASETS"'
local_path = 'data_tmp'

if not os.path.isdir(local_path):
    os.mkdir(local_path)

os.system('scp -r '+user_server+':'+remote_path+' '+local_path)
