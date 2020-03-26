# DQN
Folder for dynamic optimisation of simple fed-batch fermentation with Deep Q Networks.

## ExpDesign.py 
 - file is central integrator that runs the program

## QLearner.py 
- defines the model and target network as well as the associated hyperparameters associated with training, 
as well as the randomness of the e-greedy policy.

## Experiment_Online.py 
enables integration of decision making of the agent with dynamics of the environment and steps
environment to the next time step, over a number of episodes of experience and training rounds/epochs. Also determines
the number of weight updates per epoch. 

## Replay_Memory.py
Experience replay acts as a datastore to smooth the data distribution 

## Environment_file.py
contains definition of the dynamic bioprocess we want to dynamically optimise.


