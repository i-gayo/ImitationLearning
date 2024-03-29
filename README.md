# ImitationLearning and ReinforcementLearning for Prostate Biopsy 

An open source repository for implementing Imitation Learning (IL) and Reinforcement Learning (RL) to learn needle sample strategies for prostate biopsy procedures. 

## Conda environment creation 

To install the needed dependencies for this work, run the following command:

`conda env create -f environment.yml`

## Training: 

To train imitation learning code, run the following scripts: 

`python train_il.py`

To run RL code, run the following scripts:

`python train_rl.py`

## Evaluation: 

To evaluate the performance of the trained models (either IL or RL), run the following scripts: 

`python evaluate_models.py`

