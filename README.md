# ATTPC Energy Estimator

=======
## June 25, 2024


## Overview
In this repo there are 2 main componenets:
###
1) Simulator for particle collison in the ATTPC.
2) Initial Energy estimator for the target particle.


## Simulator
The simulator config is found in [Tools/simulation_config.yaml](Tools/simulation_config.yaml). To run the simulator, after setting config, run [Tools/utils.py](Tools/utils.py).


### Quick Simulation Guide

#### Generating Dataset
In [Tools/simulation_config.yaml](Tools/simulation_config.yaml)
* set **mode** to "generate_dataset"
* set dataset name in **dataset_name**
* set where to save the dataset in **output_dir**
* set path to angle/energy table in **angle_energy_table_path**. First column is the angle in rads and second column energy in MeV

Lastly, run [Tools/utils.py](Tools/utils.py)



### Important Adjustable Fields in Simulation Config

All the configurations are found in [Tools/simulation_config.yaml](Tools/simulation_config.yaml)
* **max_traj_length** - max length of the trajectories generated
* **stopping_power_table_path** - path to the table of energy loss (including straggling) as a function of energy.
* delta_t - time between each generated point [ns]
* **charge_std_x_axis** - STD of diffusion on the pad in the x axis
* **charge_std_y_axis** - STD of diffusion on the pad in the y axis
* **magnetic_field** - Magnitude of magnetic field [T]
* **electric_field** - Magnitude of electrical field [V/m]
* **gas_density** - density of the gas [mg/cm3 @ 1 bar]
* sensor_sampling_rate_Mhz - sensor samping rate [Mhz]
* **chamber_length** - physical length of the chamber [cm]
* **small_z_spacing** - pad's geometry
* **small_tri_side** - pad's geometry
* **umega_radius** - pad's geometry
* **CoM_observations** - if set to true, observations will be taken from a CoM of the energy cloud. Otherwise, closest pad from the single particle (cloud not used)
* **mode** - Mode of the simulation. If 'generate_dataset' is set, it will generate a dataset (multiple trajectories) if 'generate_traj' is set it will generate 1 trajectory. Configurations of both are detailed below

##### Dataset Configs
* **dataset_name** - name which the dataset will be saved under
* **output_dir** - output directory to save the dataset
* **angle_energy_table_path** - path to the table that has the energy/angle pairs to generate. The phi angle of each trajectory is uniformly chosen.
* **num_train_traj** - # of trajectories in train set
* **num_val_traj** - # of trajectories in val set
* **num_test_traj** - # of trajectories in test set
* **sub_sample_rate_data** - rate of sampling in the angle/energy table. E.g, if set to 2, it will only read every second line from the table.

##### Trajectory Configs
* **energy** - initial energy of the particle [MeV]
* **theta** - initial theta angle of the particle [rad]
* **phi** - initial phi angle of the particle
* **plot_traj** - bool flag if to plot trajectory. If false all following flags are also false
* **plot_real_traj** - bool flag if to plot real trajectory
* **plot_observed_traj** - bool flag if to plot observed trajectory
* **plot_energy_on_pad** - bool flag if to plot energy on pad

** To change the particle see the defines at the top of [Tools/utils.py](Tools/utils.py) (DRIFT_VELOCITY_CM_US , ATOMIC_NUMBER , MASS_PROTON_AMU , MASS_PROTON_KG , Q_PROTON) - currently the particle is a proton.


#### Data Format
Each Dataset consists of a list of the class Trajectory found in [Tools/utils.py](Tools/utils.py). The following are important members of the class:
* **y** - observations
* **generated_traj** - SS vectors with delta_t time stamps. Only needed for training
* **x_real** - SS vectors that are closest to the observations (XYZ sense). Only needed for training
* **t** - time steps of generated_traj that are the real_traj. E.g, if t[0] = 40, then the index 0 of the real_traj is time stamp 40 of generated_traj
* **traj_length** - length of the trajetory
* **x_estimated_FW** - SS vectors after FW pass
* **x_estimated_BW** - SS vectors after BW pass
* **BiRNN_output** - BiRNN's estimation of inital velocities. **IMPORTANT** BiRNN is optimized to give best energy estimation, the velocities arent guaranteed to be the best estimation
* **init_energy** - init energy used to generate the traj
* **init_teta** - init theta angle sued to generate the traj
* **init_phi** - init phi used to generate traj

## Energy Estimator
To estimator the energy, first the particle's XYZ (observations) are used in a RTSNet. After which the state space vectors of the particle is used in a BiRNN to generate the energy.

The config file is [Simulations/Particle_Tracking/config.yaml](Simulations/Particle_Tracking/config.yaml) and to run the estimator (whether to train or inference) run [main_Particle_Tracking.py](main_Particle_Tracking.py).

**IMPORTANT** - For inference path, it uses GT for analysis of the performance. Therefore, for data with no GT, code is needed to skip this part.


### Quick Estimator Guide

#### Training
In [Simulations/Particle_Tracking/config.yaml](Simulations/Particle_Tracking/config.yaml)
* Set **train** , **train_RTS** , and **train_BiRNN** to True 
* Add path to train/val set under  **Dataset_path** 
* Add path to test set with under **test_set_path**

Lastly, run [main_Particle_Tracking.py](main_Particle_Tracking.py).

#### Inference 
In [Simulations/Particle_Tracking/config.yaml](Simulations/Particle_Tracking/config.yaml)
* Set **train** to False
* Add path to test set with under **test_set_path**

Lastly, run [main_Particle_Tracking.py](main_Particle_Tracking.py).

### Important Adjustable Fields in Config 
All the configurations are found in [Simulations/Particle_Tracking/config.yaml](Simulations/Particle_Tracking/config.yaml)
* **state_vector_size** - size of space state vector
* **observation_vector_size** - size of observation vector
* **train_set_size** - max size of train, currently set to a # a lot higher than actual train size
* **CV_set_size** - max size of val, currently set to a # a lot higher than actual val size
* **test_set_size** - max size of test, currently set to a # a lot higher than actual test size
* **batch_size** - size of batch, currently set to a # a lot higher than actual train size, which means all train is used in each epoch
* **train** - bool flag if to train, if set to false, following two parameters are ignored and set to false
* **train_RTS** - bool flag if to train the RTSNet
* **train_BiRNN** - bool flag if to train the BiRNN
* **max_length** - max length of trajectory which the estimator will use.
* **RTS_model_path** - path to RTS model for inference, if not set , will use  Simulations/Particle_Tracking/temp models/best-model-weights_FINAL.pt
* **BiRNN_model_path** - path to BiRNN model for inference, if not set will use Simulations/Particle_Tracking/temp models/best-BiRNN_model_FINAL.pt

* **path_results** - where to save output.
* **Dataset_path** - path to train/val dataset/
* **test_set_path** - path to test dataset.

#### Output
At the end of each run, the outputs will be saved to **path_results** under a folder with the time stamp (at the end of the run the exact folder name will also be printed). The following is saved:
* All the models under the sub-dir 'models'
* Plots & CSV with the errors and energy estimation in sub-dir 'results'
* Log of the run named 'logger.log'
* A copy of the config file for the specific run
* dataset with updated smoothed SS for each trajectory and the BiRNN output, which outputs the initial velocity. To compute the energy from the velocity use get_energy_from_velocities function from [Tools/utils.py](Tools/utils.py). IMPORTANT - The net was trained to optimize energy estimation, therefore the velocities are not guaranteed to be the best estimation.






