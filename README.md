# ImitationLearning
Repo for imitation learning code used for biopsy procedures 

## The following scripts are to be used in order: 
1. Run write_labels.py to obtain labels to be used for imitation learning strategy 
2. Run train_script.py to train imitation learning network 

## Script contents 
- writing_labels.py (IL) : obtains paired observation-action labels at each time step 
- utils.py (IL) : helper functions for training networks for imitation learning 
- networks.py (IL) : UNet and Imitation learning networks to be used for training 
- train_script.py (IL) : used to train imitation learning networks based on RL-defined networks 
- train_script_timestep.py (IL) : previous script used for training, using networks defined in networks.py file 

- Biospy_env.py (RL) : environment used for RL 
- Prostate_adataloader (RL) : dataloader used to extract prostate datasets for RL environment construction 
- rl_utils.py (RL) : helper functions for training RL networks 
