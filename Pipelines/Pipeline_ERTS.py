"""
This file contains the class Pipeline_ERTS, 
which is used to train and test RTSNet in both linear and non-linear cases.
"""
from RTSNet.RTSNet_nn import RTSNetNN
import torch
import torch.nn as nn
import time
import random
import numpy as np
from Plot import Plot_extended as Plot
from Tools.utils import get_mx_0,System_Mode,Trajectory_SS_Type,estimation_summary,get_energy_from_velocities,get_velocity_from_energy,spherical_to_cartersian_co,SS_VARIABLE
import os
import matplotlib.pyplot as plt
import time
from datetime import datetime
import shutil
import stat

class Pipeline_ERTS:

    def __init__(self, Time, folderName, modelName,system_config):
        super().__init__()
        self.config = system_config
        self.Time = Time
        self.modelName = modelName
        self.phase_change_epochs = [0] #for Visualization - aggregate the epochs which a phase changed
        if self.config.use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def setssModel(self, ssModel):
        self.SysModel = ssModel

    def setModel(self, model):
        self.model = model
        self.model.config = self.config

    def setHeadPipeline(self,pipeline):
        self.head_pipeline = pipeline

    def setTrainingParams(self):
        self.batch_size = self.config.batch_size # Number of tracks in Batch
        self.total_num_phases = len(self.config.training_scheduler)

        self.current_phase = str(self.config.first_phase_id)
        self.num_epochs = np.sum([phase_config["n_epochs"] for phase_id,phase_config in enumerate(self.config.training_scheduler.values()) if phase_id>=self.config.first_phase_id])  # Number of Training Steps

        if self.current_phase !="0":
            prev_phase = str(int(self.current_phase)-1)
            weights_path = os.path.join(self.config.path_results,"temp models",f"best-model-weights_P{prev_phase}.pt")
            model_weights = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(model_weights)
            self.logger.info(f"Loaded Weights from {os.path.basename(weights_path)}")
            # self.head_pipeline.model.load_state_dict(torch.load(os.path.join(self.config.path_results,"temp models",f"best-BiRNN_model_init.pt"),map_location=self.device))
        self.learningRate = self.config.training_scheduler[self.current_phase]["lr"] # Learning Rate of first phase
        self.num_epochs_in_phase = self.config.training_scheduler[self.current_phase]["n_epochs"]
        self.next_phase_change = self.num_epochs_in_phase # Which epoch to change to next phase
        self.SYSTEM_MODE = System_Mode(self.config.training_scheduler[self.current_phase]["mode"])
        self.phase_modes = [self.SYSTEM_MODE.value] #For visualization - set first mode type
        self.spoon_feeding = self.config.training_scheduler[self.current_phase]["spoon_feeding"] if "spoon_feeding" in self.config.training_scheduler[self.current_phase] else 0
        self.loss = self.config.training_scheduler[self.current_phase]["loss"] if "loss" in self.config.training_scheduler[self.current_phase] else 'all'
        self.weightDecay = self.config.wd # L2 Weight Regularization - Weight Decay
        self.loss_fn = nn.MSELoss(reduction='mean')
        self.set_optimizer()
        if self.config.train == True:
            self.report_training_phase()

    def set_optimizer(self):
        # Use the optim package to define an Optimizer that will update the weights of
        # the model for us. Here we will use Adam; the optim package contains many other
        # optimization algoriths. The first argument to the Adam constructor tells the
        # optimizer which Tensors it should update.

        if self.SYSTEM_MODE == System_Mode.FW_BW:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learningRate, weight_decay=self.weightDecay)
        elif self.SYSTEM_MODE == System_Mode.FW_ONLY:
            self.optimizer = torch.optim.Adam(self.model.KNET_params, lr=self.learningRate, weight_decay=self.weightDecay)
        elif self.SYSTEM_MODE == System_Mode.BW_ONLY:
            self.optimizer = torch.optim.Adam(self.model.RTSNET_params, lr=self.learningRate, weight_decay=self.weightDecay)
        elif self.SYSTEM_MODE == System_Mode.BW_HEAD:
            self. optimizer = torch.optim.Adam([
                    {'params': self.model.RTSNET_params, 'lr': self.learningRate},
                    {'params': self.head_pipeline.model.BiRNN_param, 'lr': self.config.BiRNN_lr}], weight_decay=self.weightDecay)
        elif self.SYSTEM_MODE == System_Mode.FW_BW_HEAD:
             self. optimizer = torch.optim.Adam([
                    {'params': self.model.RTSNET_params, 'lr': self.learningRate},
                    {'params': self.head_pipeline.model.BiRNN_param, 'lr': self.config.BiRNN_lr},
                    {'params': self.model.KNET_params, 'lr': self.learningRate}], weight_decay=self.weightDecay)


    def switch_to_next_phase(self):
        #Load best weights from phase
        weights_path = os.path.join(self.config.path_results,"temp models",f"best-model-weights_P{self.current_phase}.pt")
        model_weights = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(model_weights)
        #continue to next phase
        self.current_phase = str(int(self.current_phase)+1)
        self.SYSTEM_MODE = System_Mode(self.config.training_scheduler[self.current_phase]["mode"])
        self.num_epochs_in_phase = self.config.training_scheduler[self.current_phase]["n_epochs"]
        self.spoon_feeding = self.config.training_scheduler[self.current_phase]["spoon_feeding"] if "spoon_feeding" in self.config.training_scheduler[self.current_phase] else False
        self.loss = self.config.training_scheduler[self.current_phase]["loss"] if "loss" in self.config.training_scheduler[self.current_phase] else 'all'
        self.next_phase_change = self.next_phase_change + self.num_epochs_in_phase
        self.learningRate = self.config.training_scheduler[self.current_phase]["lr"]
        self.set_optimizer() #updates learning rate/training weights
        self.report_training_phase()

    def report_training_phase(self):
        self.logger.info(f"\n######## Entered Training Phase {self.current_phase} {self.modelName} ########\n\n"
              f"Num. of epochs in phase : {self.num_epochs_in_phase}\n"
              f"Next Switch Epoch : {self.next_phase_change}\n"
              f"Learning Rate : {self.learningRate}\n"
              f"Mode : {self.SYSTEM_MODE.value}\n"
              f"Loss : {self.loss}\n"
              f"Spoon Feeding: {self.spoon_feeding}\n"
              f"Training {sum(p.numel() for group in self.optimizer.param_groups for p in group['params'] if p.requires_grad)} parameters\n\n"
              f"######## Entered Training Phase {self.current_phase} {self.modelName} ########\n")

    def NNTrain(self, SysModel, train_set, cv_set , run_num):

        assert len(cv_set), "CV set size is 0!"
        assert len(train_set), "CV set size is 0!"

        self.train_set_size = len(train_set)
        self.CV_set_size = len(cv_set)


        self.MSE_cv_linear_epoch = torch.zeros([self.num_epochs])
        self.MSE_cv_dB_epoch = torch.zeros([self.num_epochs])
        self.MSE_cv_opt_id_phase = torch.zeros([self.total_num_phases])
        self.MSE_cv_opt_dB_phase = torch.zeros([self.total_num_phases])

        self.MSE_train_linear_epoch = torch.zeros([self.num_epochs])
        self.MSE_train_dB_epoch = torch.zeros([self.num_epochs])

        ##############
        ### Epochs ###
        ##############

        self.MSE_cv_dB_opt = 1000
        self.MSE_cv_idx_opt = 0     
        force_FW_pass = True #Always on first epoch perform FW pass
        for ti in range(0, self.num_epochs):
            start = time.time()
            if ti == self.next_phase_change:
                self.MSE_cv_opt_dB_phase[int(self.current_phase)] = self.MSE_cv_dB_opt
                self.MSE_cv_opt_id_phase[int(self.current_phase)] = self.MSE_cv_idx_opt
                self.MSE_cv_dB_opt = 1000
                self.phase_change_epochs.append(ti) #for Visualization - aggregate the epochs which a phase changed
                self.switch_to_next_phase()
                self.phase_modes.append(self.SYSTEM_MODE.value)#for Visualization
                force_FW_pass = True #When switching phase, on first epoch do fw pass

            ###############################
            ### Training Sequence Batch ###
            ###############################

            # Training Mode
            self.model.train()
            self.head_pipeline.model.train()
            self.optimizer.zero_grad()

            # Randomly select N_B training sequencesgit stat
            self.batch_size = min(self.batch_size, self.train_set_size) # N_B must be smaller than N_E
            n_e = self.config.force_batch if len(self.config.force_batch) else random.sample(range(self.train_set_size), k=self.batch_size)
            train_batch = [train_set[idx] for idx in n_e if train_set[idx].traj_length>-1]

            do_fw_pass = force_FW_pass or "FW" in self.SYSTEM_MODE.value #Do FW Pass when we back prop FW pass or when first epoch of a phase
            MSE_train_batch_linear_LOSS = self.calculate_loss(train_batch,SysModel,do_FW_pass=do_fw_pass,run_num=run_num)
            # dB Loss
            self.MSE_train_linear_epoch[ti] = MSE_train_batch_linear_LOSS.item()
            self.MSE_train_dB_epoch[ti] = 10 * torch.log10(self.MSE_train_linear_epoch[ti])

            ##################
            ### Optimizing ###
            ##################

            # Before the backward pass, use the optimizer object to zero all of the
            # gradients for the variables it will update (which are the learnable
            # weights of the model). This is because by default, gradients are
            # accumulated in buffers( i.e, not overwritten) whenever .backward()
            # is called. Checkout docs of torch.autograd.backward for more details.

            # Backward pass: compute gradient of the loss with respect to model
            # parameters
        
            # prev_KNET_weights = [param.clone() for param in self.model.KNET_params]
            # prev_RTS_weights = [param.clone() for param in self.model.RTSNET_params]
            # prev_BiRNN_weights = [param.clone() for param in self.head_pipeline.model.parameters()]

            MSE_train_batch_linear_LOSS.backward(retain_graph=True)
            self.optimizer.step()

            # knet_updated = any(
            # not torch.equal(prev, current)
            # for prev, current in zip(prev_KNET_weights, self.model.KNET_params)
            # )

            # rtsnet_updated = any(
            #     not torch.equal(prev, current)
            #     for prev, current in zip(prev_RTS_weights, self.model.RTSNET_params)
            # )

            # BiRNN_updated = any(
            #     not torch.equal(prev, current)
            #     for prev, current in zip(prev_BiRNN_weights, self.head_pipeline.model.parameters())
            # )

            # print(f"Knet Updated {knet_updated} , RTS Updated {rtsnet_updated} , BiRNN Updated {BiRNN_updated}")

            # self.scheduler.step(self.MSE_cv_dB_epoch[ti])
            #################################
            ### Validation Sequence Batch ###
            #################################
            # Cross Validation Mode
            self.model.eval()
            self.head_pipeline.model.eval()
            with torch.no_grad():
                do_fw_pass = force_FW_pass or "FW" in self.SYSTEM_MODE.value #Do FW Pass when we back prop FW pass or when first epoch of a phase
                MSE_cv_linear_LOSS = self.calculate_loss(cv_set,SysModel,do_FW_pass=do_fw_pass,run_num=run_num)

                # dB Loss
                self.MSE_cv_linear_epoch[ti] = MSE_cv_linear_LOSS.item()
                self.MSE_cv_dB_epoch[ti] = 10 * torch.log10(self.MSE_cv_linear_epoch[ti])

                if (self.MSE_cv_dB_epoch[ti] < self.MSE_cv_dB_opt or ti==0):
                    self.MSE_cv_dB_opt = self.MSE_cv_dB_epoch[ti]
                    self.MSE_cv_idx_opt = ti
                    best_model = self.model.state_dict().copy()
                    if "HEAD" in self.SYSTEM_MODE.value:
                        best_BiRNN_model = self.head_pipeline.model.state_dict().copy()
                    
                    torch.save(self.model.state_dict(), os.path.join(self.config.path_results,"temp models",f"best-model-weights_P{self.current_phase}.pt"))
                    if "HEAD" in self.SYSTEM_MODE.value:
                        torch.save(self.head_pipeline.model.state_dict(), os.path.join(self.config.path_results,"temp models",f"best-BiRNN_model_P{self.current_phase}.pt"))
                    if int(self.current_phase) == self.total_num_phases-1:
                        torch.save(self.model.state_dict(), os.path.join(self.config.path_results,"temp models",f"best-model-weights_FINAL.pt"))
                        if "HEAD" in self.SYSTEM_MODE.value: 
                            torch.save(self.head_pipeline.model.state_dict(), os.path.join(self.config.path_results,"temp models",f"best-BiRNN_model_FINAL.pt"))
            ########################
            ### Training Summary ###
            ########################
            force_FW_pass = False #This will only be set true again at the beginning of the next Phase
            self.logger.info(f"Total Time : {time.time()-start}")
            self.logger.info(f"P{self.current_phase} {ti} MSE Training : {round(self.MSE_train_dB_epoch[ti].item(),3)} [dB] MSE Validation : {round(self.MSE_cv_dB_epoch[ti].item(),3)} [dB]")

            if (ti > 0):
                d_train = self.MSE_train_dB_epoch[ti] - self.MSE_train_dB_epoch[ti - 1]
                d_cv = self.MSE_cv_dB_epoch[ti] - self.MSE_cv_dB_epoch[ti - 1]
                self.logger.info(f"diff MSE Training : {round(d_train.item(),3)} [dB] diff MSE Validation : {round(d_cv.item(),3)} [dB]")
            self.logger.info(f"Optimal idx: {self.MSE_cv_idx_opt} Optimal : {round(self.MSE_cv_dB_opt.item(),3)} [dB]\n")

        torch.save(best_model, os.path.join(self.config.path_results,"temp models",f"best-model-weights_FINAL.pt"))
        if "HEAD" in self.SYSTEM_MODE.value:
            torch.save(best_BiRNN_model, os.path.join(self.config.path_results,"temp models",f"best-BiRNN_model_FINAL.pt"))

        return [self.MSE_cv_linear_epoch, self.MSE_cv_dB_epoch, self.MSE_train_linear_epoch, self.MSE_train_dB_epoch]

    def plot_training_summary(self,output_path,run_num):
        self.MSE_cv_opt_dB_phase[int(self.current_phase)] = self.MSE_cv_dB_opt
        self.MSE_cv_opt_id_phase[int(self.current_phase)] = self.MSE_cv_idx_opt
        plt.figure()
        plt.plot(range(len(self.MSE_train_dB_epoch)),self.MSE_train_dB_epoch,label='Train loss',linewidth=0.5)
        plt.plot(range(len(self.MSE_cv_dB_epoch)),self.MSE_cv_dB_epoch,label='CV loss',linewidth=0.8,linestyle='--')
        plt.scatter(self.MSE_cv_opt_id_phase[:int(self.current_phase) + 1], self.MSE_cv_opt_dB_phase[:int(self.current_phase)+1], color='green', marker='o',s=10,label='Opt MSE')  # Mark specific points
        # # # Mark best MSE in each phase
        # for i in range(min(len(self.phase_change_epochs),len(self.MSE_cv_opt_dB_phase))):
        #     plt.text(self.phase_change_epochs[i] + 20, plt.gca().get_ylim()[1] + 0.5, f'Opt MSE {round(self.MSE_cv_opt_dB_phase[i].item(),2)})', fontsize=10)
        # Mark Phase switches
        for epoch in self.phase_change_epochs:
            plt.axvline(x=epoch, color='red', linestyle='--',linewidth=0.7)
        for phase_id,epoch in enumerate(self.phase_change_epochs):
            plt.text(epoch, plt.gca().get_ylim()[1] + 0.7, f'P{phase_id}\n ({self.phase_modes[phase_id]})', ha='center')
            self.logger.info(f"Optimal CV MSE Phase {phase_id + self.config.first_phase_id} : {round(self.MSE_cv_opt_dB_phase[phase_id].item(),2)} [dB]")

        plt.xlabel("Epoch")
        plt.ylabel('MSE [dB]')
        plt.title("Training Losses", y=1.09)
        plt.legend()
        plt.savefig(os.path.join(output_path,f"Learning_Curve_.png"))

    def NNEval(self, SysModel, eval_set,set_name,load_RTS_model_path=None,load_BiRNN_model_path=None):
        # Wrapper for NNtest with deep unfolding
        self.test_set_size = self.model.batch_size = len(eval_set)
        today = datetime.today()
        now = datetime.now()
        run_path = os.path.join(self.config.path_results,"runs",f"{today.strftime('D%d_M%m')}_{now.strftime('h%H_m%M')}") if set_name == "Test" else None
        self.SYSTEM_MODE = System_Mode.FW_BW
        self.loss = self.config.test_loss
        if load_RTS_model_path is not None:
            model_weights = torch.load(load_RTS_model_path, map_location=self.device) 
        else:
            load_RTS_model_path = os.path.join(self.config.path_results,"temp models",f'best-model-weights_FINAL.pt')
        
        with open(load_RTS_model_path, 'rb') as f:
            model_weights = torch.load(f, map_location=self.device)
        self.model.load_state_dict(model_weights)

        self.logger.info(f"\n######## Entered Eval {self.modelName} ########\n\n"
              f"Set : {set_name}\n"
              f"Mode : {self.SYSTEM_MODE.value}\n"
              f"Loss : {self.loss}\n"
              f"RTS Model : {load_RTS_model_path}\n"
              f"run path : {run_path}\n\n"
              f"######## Entered Eval  {self.modelName} ########")

        prev_loss = float('inf')
        run_num=0
        while True:
            print(f"\n -- run {run_num} -- ")
            self.NNTest(SysModel,test_set=eval_set,run_num = run_num ,set_name=set_name,load_BiRNN_model_path=load_BiRNN_model_path,run_path=run_path)
            if self.MSE_test_RTS_linear_avg >= prev_loss or not(self.config.deep_unfolding):
                break
            prev_loss = self.MSE_test_RTS_linear_avg
            run_num +=1
        return run_path

    def NNTest(self, SysModel, test_set,run_num,set_name,load_BiRNN_model_path,run_path):


        # Test mode
        self.model.eval()
        # self.head_pipeline.model.eval()
        # Init Hidden State
        self.model.init_hidden()
        with torch.no_grad():
            if run_num == 0 or self.loss == 'all': 
                set = test_set
            else:
                #rerun only on low energies 
                set_indices = [i for i, traj in enumerate(test_set) if self.config.max_energy_to_rerun > get_energy_from_velocities(traj.initial_state_estimation[-1][-3],
                                                                                                                   traj.initial_state_estimation[-1][-2],
                                                                                                                   traj.initial_state_estimation[-1][-1])]
                set = [test_set[i] for i in set_indices]
            start = time.time()
            self.MSE_test_RTS_linear_avg = self.calculate_loss(set,SysModel,do_FW_pass=True,run_num=run_num)
            self.MSE_test_BiRNN_linear_avg = None

        for traj in test_set:
            traj.initial_state_estimation = traj.x_estimated_BW[:,0]

        if set_name == "Test":
            #Run Head
            run_models_path = os.path.join(run_path,"models")
            run_results_path = os.path.join(run_path,"results")
            os.makedirs(run_models_path,exist_ok=True)
            os.makedirs(run_results_path,exist_ok=True)
            shutil.copytree(os.path.join(self.config.path_results, "temp models"), run_models_path,dirs_exist_ok=True)
            if not(os.path.exists(os.path.join(run_path,"run_config_readonly.yaml"))):
                shutil.copyfile("Simulations/Particle_Tracking/config.yaml", os.path.join(run_path,"run_config_readonly.yaml"))
                os.chmod(os.path.join(run_path,"run_config_readonly.yaml"), stat.S_IREAD)
            try:
                self.MSE_test_BiRNN_linear_avg = self.head_pipeline.eval(test_set,load_BiRNN_model_path)
                self.head_pipeline.plot_data(test_set,run_results_path)
            except Exception as e:
                self.logger.info(f"BiRNN Eval Failed!\n{e}")
            estimation_summary(test_set,run_results_path,run_num)
            try:
                self.plot_training_summary(run_results_path,run_num)
            except:
                self.logger.info("plot_training_summary Failed")

        end = time.time()
        t = end - start
        if self.MSE_test_BiRNN_linear_avg is not None:
            self.logger.info(f"Final Eval {self.modelName} - MSE {set_name}: Total (lambda = {self.config.lambda_loss}): {10 * torch.log10(self.MSE_test_BiRNN_linear_avg + self.config.lambda_loss * self.MSE_test_RTS_linear_avg).item()}[dB] , RTS {10 * torch.log10(self.MSE_test_RTS_linear_avg).item()}[dB] , BiRNN {10 * torch.log10(self.MSE_test_BiRNN_linear_avg).item()} [dB]")
        else:
            self.logger.info(f"Final Eval {self.modelName} - MSE {set_name}: RTS {10 * torch.log10(self.MSE_test_RTS_linear_avg).item()}[dB]")

        self.logger.info("Inference Time: %d", t)
        
        if set_name == "Test":
            shutil.copy2(os.path.join(self.config.path_results,"temp_log.log"), os.path.join(run_path,'logger.log'))

        return run_path

    def calculate_loss(self,traj_batch,SysModel,do_FW_pass,run_num):
            
            self.model.batch_size = len(traj_batch)
            # Init Hidden State
            self.model.init_hidden()
            self.config.delta_t = torch.ones([self.model.batch_size]) * self.config.FTT_delta_t #forward pass has the finest Delta

            clustered_traj_lengths_in_batch = torch.tensor([min(self.config.max_length,traj.x_real.shape[1]) for traj in traj_batch])
            generated_traj_lengths_in_batch = torch.tensor([traj.generated_traj.shape[1] for traj in traj_batch])
            max_clustered_traj_length_in_batch = torch.max(clustered_traj_lengths_in_batch)
            max_generated_traj_length_in_batch = torch.max(generated_traj_lengths_in_batch) * 2

            ## Init Training Batch tensors##
            #lengths before imputation
            batch_y = torch.zeros([len(traj_batch), SysModel.observation_vector_size, max_clustered_traj_length_in_batch])

            #lengths after imputation
            batch_target = torch.zeros([len(traj_batch), SysModel.space_state_size, max_generated_traj_length_in_batch])
            fw_output_place_holder = torch.zeros([len(traj_batch), SysModel.space_state_size, max_generated_traj_length_in_batch])
            x_out_forward_batch = torch.zeros([len(traj_batch), SysModel.space_state_size, max_generated_traj_length_in_batch])
            x_out_batch = torch.zeros([len(traj_batch), SysModel.space_state_size, max_generated_traj_length_in_batch])
            clustered_in_generated_mask = torch.zeros([len(traj_batch), SysModel.space_state_size, max_generated_traj_length_in_batch],dtype=torch.bool)#if index is True , that SS is GTT and FTT -- for FW Pass Loss
            update_step_in_fw_mask = torch.zeros([len(traj_batch), SysModel.space_state_size, max_generated_traj_length_in_batch],dtype=torch.bool)# Mark which time steps in FW had update step from KF -- for FW Pass Loss
            steps_to_smooth_map = torch.zeros([len(traj_batch), max_generated_traj_length_in_batch],dtype=torch.int) + -1# Saves IDs of update steps to perform BW
            bw_gain = torch.zeros([len(traj_batch),SysModel.space_state_size**2, max_generated_traj_length_in_batch],dtype=torch.float) + -1# Saves IDs of update steps to perform BW
            fw_gain = torch.zeros([len(traj_batch),SysModel.space_state_size * SysModel.observation_vector_size, max_generated_traj_length_in_batch],dtype=torch.float) + -1# Saves IDs of update steps to perform BW
            bw_inov = torch.zeros([len(traj_batch),SysModel.space_state_size, max_generated_traj_length_in_batch],dtype=torch.float) + -1# Saves IDs of update steps to perform BW
            bw_dx = torch.zeros([len(traj_batch),SysModel.space_state_size, max_generated_traj_length_in_batch],dtype=torch.float) + -1# Saves IDs of update steps to perform BW

            #If we are not backproping through forward pass, we dont need to redo FW pass - just take saved pass. BATCH SIZE must be equal to SET SIZE
            if do_FW_pass:
                M1_0 = []
                for ii in range(len(traj_batch)):
                    batch_y[ii,:,:clustered_traj_lengths_in_batch[ii]] = traj_batch[ii].y[:3,:clustered_traj_lengths_in_batch[ii]].squeeze(-1) if run_num == 0 else traj_batch[ii].x_estimated_BW[:3,:clustered_traj_lengths_in_batch[ii]].squeeze(-1)
                    batch_target[ii,:,:generated_traj_lengths_in_batch[ii]] = traj_batch[ii].generated_traj[:,:generated_traj_lengths_in_batch[ii]].squeeze(-1)
                    clustered_in_generated_mask[ii,:,traj_batch[ii].t[1:clustered_traj_lengths_in_batch[ii]]] = 1 #The first t is M1_0,
                    energy_at_first_cluster = get_energy_from_velocities(traj_batch[ii].x_real[3,0],traj_batch[ii].x_real[4,0],traj_batch[ii].x_real[5,0])
                    m1,est_para = get_mx_0(traj_batch[ii].y.squeeze(-1),energy_at_first_cluster=energy_at_first_cluster,use_traj_for_energy=True,error_perc=10)
                    # m1 = traj_batch[ii].BiRNN_output #traj_batch[ii].x_real[:,0].squeeze(-1) *1.0001 #traj_batch[ii].BiRNN_output #
                    # if hasattr(traj_batch[ii], 'initial_state_estimation'):
                    #     m1[3:] = traj_batch[ii].initial_state_estimation[3:].reshape(-1)
                    M1_0.append(m1.unsqueeze(0))
                    ii += 1

                M1_0 = torch.cat(M1_0,dim=0)
                x_out_forward_batch[:, :, 0] = M1_0
                self.model.InitSequence(M1_0.unsqueeze(-1),max_clustered_traj_length_in_batch)
                # Forward Computation
                fine_step_for_each_trajID_in_batch = torch.zeros(len(traj_batch),dtype=torch.int)
                for t in range(1,max_clustered_traj_length_in_batch):

                    distance_from_obs_for_each_trajID_in_batch = torch.full((len(traj_batch),), 1e5)
                    traj_id_in_batch_finished = t > (clustered_traj_lengths_in_batch - 1) #since we run till the maximum t in all trajs in batch, some might end before the others
                    traj_id_in_batch_that_need_prediction = ~traj_id_in_batch_finished
                    xt_minus_1 = torch.zeros([len(traj_batch),SysModel.space_state_size,1])
                    #predict until we get to the next observation. Prediction is only via the propagation function, no use of RNN
                    while(any(traj_id_in_batch_that_need_prediction)):

                        #init
                        new_distances_to_obs =  torch.full((len(traj_batch),), 1e5)
                        next_ss_via_prediction_only = torch.zeros([len(traj_batch),SysModel.space_state_size])

                        # perform prediction to relevant trajs
                        with torch.no_grad():
                            next_ss_via_prediction_only[traj_id_in_batch_that_need_prediction] = self.model.f(x_out_forward_batch[traj_id_in_batch_that_need_prediction,:,fine_step_for_each_trajID_in_batch[traj_id_in_batch_that_need_prediction]],self.config.delta_t[traj_id_in_batch_that_need_prediction]).squeeze(-1)

                        # get distance of predictions to next observations
                        new_distances_to_obs[traj_id_in_batch_that_need_prediction] = torch.sqrt(torch.sum((next_ss_via_prediction_only[traj_id_in_batch_that_need_prediction,:3] - batch_y[traj_id_in_batch_that_need_prediction,:3,t])**2,dim=1))
                        # print(f"Est : {next_ss_via_prediction_only[traj_id_in_batch_that_need_prediction][0,:3]} , Obs {batch_y[traj_id_in_batch_that_need_prediction,:3,t]} , New Dist {new_distances_to_obs} Old Dis{distance_from_obs_for_each_trajID_in_batch}")
                        # Mark which trajs are still getting closer to their obs and which have passed it.
                        # If it doesnt need predicition it will get a False on "getting_closer" and we also add a False in "getting farther"
                        trajs_getting_closer = new_distances_to_obs < distance_from_obs_for_each_trajID_in_batch
                        traj_getting_farther = ~trajs_getting_closer & traj_id_in_batch_that_need_prediction

                        #########################################
                        ## For those who we are getting closer ##
                        #########################################
                        # Update SS in forward batch
                        # Increment fine step
                        # Update new distance from obs
                        time_stamp_to_update = torch.min(fine_step_for_each_trajID_in_batch[trajs_getting_closer]+1,max_generated_traj_length_in_batch-1) # against overflow
                        x_out_forward_batch[trajs_getting_closer,:,time_stamp_to_update] = next_ss_via_prediction_only[trajs_getting_closer]
                        fine_step_for_each_trajID_in_batch[trajs_getting_closer]= time_stamp_to_update
                        distance_from_obs_for_each_trajID_in_batch[trajs_getting_closer] = new_distances_to_obs[trajs_getting_closer]

                        if any(time_stamp_to_update == max_generated_traj_length_in_batch-1):
                            self.logger.info("Overflow in prediction!")

                        ###############################################
                        ## For those who we are getting farther away ##
                        ###############################################
                        # Mark no more predictions needed
                        # Mark that this time step will be getting an update step in KNET
                        # We want to Knet to Predict from fine_step_for_each_trajID_in_batch[traj_id_in_batch]-1 and we update the results of Knet to fine_step_for_each_trajID_in_batch[traj_id_in_batch]
                        traj_id_in_batch_that_need_prediction[traj_getting_farther] = False
                        update_step_in_fw_mask[traj_getting_farther,:,fine_step_for_each_trajID_in_batch[traj_getting_farther]] = 1
                        xt_minus_1[traj_getting_farther,:,:] = x_out_forward_batch[traj_getting_farther,:,fine_step_for_each_trajID_in_batch[traj_getting_farther]-1].unsqueeze(-1)
                
                    output = torch.squeeze(self.model(yt = (batch_y[:, :, t].unsqueeze(-1)),xt_minus_1=xt_minus_1),-1) #Model Expects [num_batches,obs_vector_size,1]
                    if self.SYSTEM_MODE == System_Mode.FW_ONLY and self.spoon_feeding: #Till the KNET warms up a bit
                        fw_output_place_holder[~traj_id_in_batch_finished,:,fine_step_for_each_trajID_in_batch[~traj_id_in_batch_finished]] = output[~traj_id_in_batch_finished]
                    else:
                        x_out_forward_batch[~traj_id_in_batch_finished,:,fine_step_for_each_trajID_in_batch[~traj_id_in_batch_finished]] = output[~traj_id_in_batch_finished]
                    fw_gain[~traj_id_in_batch_finished,:,fine_step_for_each_trajID_in_batch[~traj_id_in_batch_finished]] = self.model.KGain.reshape(len(traj_batch),-1).detach()[~traj_id_in_batch_finished,:] 
            else:#if do_FW_pass
                #Take the FW estimation from previous run. We only save points where "update" was done.
                with torch.no_grad():
                    for traj_id in range(len(traj_batch)):
                        traj_length = traj_batch[traj_id].traj_length
                        #copy the updated steps
                        x_out_forward_batch[traj_id,:,traj_batch[traj_id].t_update_step_in_FW.squeeze(-1)] = traj_batch[traj_id].x_estimated_FW[:,1:traj_length].squeeze(-1).detach()
                        #copy M1_0
                        x_out_forward_batch[traj_id,:,0] = traj_batch[traj_id].x_estimated_FW[:,0].squeeze(-1)
                        batch_target[traj_id,:,:generated_traj_lengths_in_batch[traj_id]] = traj_batch[traj_id].generated_traj[:,:generated_traj_lengths_in_batch[traj_id]].squeeze(-1)
                        update_step_in_fw_mask[traj_id,:,traj_batch[traj_id].t_update_step_in_FW.squeeze(-1)] = 1

            # Backward Computation
            if self.SYSTEM_MODE != System_Mode.FW_ONLY:
                batch_ids = torch.arange(len(traj_batch))
                est_BW_mask_loss =  torch.zeros_like(update_step_in_fw_mask.clone())
                FTT_BW_mask_loss = torch.zeros_like(clustered_in_generated_mask)

                for id_in_batch in range(len(traj_batch)):
                    cluster_ids_in_fw = torch.cat((torch.tensor([[0]]), update_step_in_fw_mask[id_in_batch,0,:].nonzero()), dim=0) #add the first cluster (this was ignored in FW)
                    cluster_ids_in_FFT = traj_batch[id_in_batch].t[:clustered_traj_lengths_in_batch[id_in_batch]]
                    relevant_ids_est = list()
                    relevant_ids_FTT = list()

                    # if self.config.interpolation_points_to_add == 0:
                    relevant_ids_est = cluster_ids_in_fw.reshape(-1).tolist()
                    relevant_ids_FTT = cluster_ids_in_FFT.reshape(-1).tolist()

                    #Ids to smooth in BW
                    steps_to_smooth_map[id_in_batch,:len(relevant_ids_est)] = torch.flip(torch.tensor(relevant_ids_est),dims=[0]).squeeze()

                    #Init Velocity Loss Mask
                    if self.loss == 'init_velocity' or self.loss == 'init_energy':
                        #Loss on the velocities of the first cluster
                        FTT_BW_mask_loss[id_in_batch] = False
                        FTT_BW_mask_loss[id_in_batch,[SS_VARIABLE.Vx.value,SS_VARIABLE.Vy.value,SS_VARIABLE.Vz.value],traj_batch[id_in_batch].t[0]] = True 
                        est_BW_mask_loss[id_in_batch] = False
                        est_BW_mask_loss[id_in_batch,[SS_VARIABLE.Vx.value,SS_VARIABLE.Vy.value,SS_VARIABLE.Vz.value],0] = True
                    if self.loss == 'init_state':
                        FTT_BW_mask_loss[id_in_batch] = False
                        FTT_BW_mask_loss[id_in_batch,:,traj_batch[id_in_batch].t[0]] = True 
                        est_BW_mask_loss[id_in_batch] = False
                        est_BW_mask_loss[id_in_batch,:,0] = True
                    elif self.loss == 'all':
                        #Loss on all points in trajectory
                        FTT_BW_mask_loss[id_in_batch,:,relevant_ids_FTT] = True 
                        est_BW_mask_loss[id_in_batch,:,relevant_ids_est] = True
                    elif self.loss == 'velocity' or self.loss == 'energy':
                        #Loss on velocity points in trajectory
                        FTT_BW_mask_loss[id_in_batch,-3:,relevant_ids_FTT] = True 
                        est_BW_mask_loss[id_in_batch,-3:,relevant_ids_est] = True
                    elif self.loss == 'pos':
                        #Loss on positional points in trajectory
                        FTT_BW_mask_loss[id_in_batch,:3,relevant_ids_FTT] = True 
                        est_BW_mask_loss[id_in_batch,:3,relevant_ids_est] = True

                    # elif self.loss == 'interpolation': 
                    #     #Loss only on interpolation points not on clusters
                    #     est_BW_mask_loss[id_in_batch] = est_BW_mask_loss[id_in_batch] & ~update_step_in_fw_mask
                    #     FTT_BW_mask_loss[id_in_batch] = FTT_BW_mask_loss[id_in_batch] & ~clustered_in_generated_mask
                    

                x_out_batch[batch_ids,:,steps_to_smooth_map[:,0]] = x_out_forward_batch[batch_ids,:,steps_to_smooth_map[:,0]] #It was flipped such that 0 in the flipped is steps_to_smooth_map[:,0]
                self.model.InitBackward(torch.unsqueeze(x_out_batch[batch_ids, :, steps_to_smooth_map[:,0]],2))
                
                self.config.delta_t =  (steps_to_smooth_map[:,0] - steps_to_smooth_map[:,1]) * self.config.FTT_delta_t
                x_out_batch[batch_ids, :, steps_to_smooth_map[:,1]] = self.model(filter_x =x_out_forward_batch[batch_ids, :, steps_to_smooth_map[:,1]].unsqueeze(2),
                                                                                filter_x_nexttime = x_out_batch[batch_ids, :, steps_to_smooth_map[:,0]].unsqueeze(2)).squeeze(2)

                for k in range(2,steps_to_smooth_map.shape[1]):
                    end_of_traj = steps_to_smooth_map[:,k] == -1
                    if torch.all(end_of_traj):
                        break
                    self.config.delta_t =  (steps_to_smooth_map[:,k-1] - steps_to_smooth_map[:,k]) * self.config.FTT_delta_t
                    x_out_batch[~end_of_traj, :, steps_to_smooth_map[~end_of_traj,k]] = torch.squeeze(self.model(filter_x = torch.unsqueeze(x_out_forward_batch[batch_ids, :, steps_to_smooth_map[:,k]],2), 
                                                                                    filter_x_nexttime = torch.unsqueeze(x_out_batch[batch_ids, :, steps_to_smooth_map[:,k-1]],2),
                                                                                    smoother_x_tplus2 = torch.unsqueeze(x_out_batch[batch_ids, :, steps_to_smooth_map[:,k-2]],2)),2)[~end_of_traj,:]
                    if torch.isnan(x_out_batch[~end_of_traj, :, steps_to_smooth_map[~end_of_traj,k]]).any():
                        print(f"NaN detected in output {k}")
                    
                    bw_gain[~end_of_traj,:,steps_to_smooth_map[~end_of_traj,k]] = self.model.SGain.reshape(len(traj_batch),-1).detach()[~end_of_traj,:]
                    bw_inov[~end_of_traj,:,steps_to_smooth_map[~end_of_traj,k]] = self.model.inov.detach().squeeze(-1)[~end_of_traj,:]
                    bw_dx[~end_of_traj,:,steps_to_smooth_map[~end_of_traj,k]] = self.model.dx.detach().squeeze(-1)[~end_of_traj,:]

            #Compute  loss
            if self.SYSTEM_MODE == System_Mode.FW_ONLY:
                est_FW_mask_loss = update_step_in_fw_mask.clone()
                FTT_FW_mask_loss = clustered_in_generated_mask.clone()
                if self.loss == 'pos':
                    est_FW_mask_loss[:,[SS_VARIABLE.Vx.value,SS_VARIABLE.Vy.value,SS_VARIABLE.Vz.value],:] = False
                    FTT_FW_mask_loss[:,[SS_VARIABLE.Vx.value,SS_VARIABLE.Vy.value,SS_VARIABLE.Vz.value],:] = False
                elif self.loss == 'velocity':
                    est_FW_mask_loss[:,[SS_VARIABLE.X.value,SS_VARIABLE.Y.value,SS_VARIABLE.Z.value],:] = False
                    FTT_FW_mask_loss[:,[SS_VARIABLE.X.value,SS_VARIABLE.Y.value,SS_VARIABLE.Z.value],:] = False
                if self.spoon_feeding: #Till the KNET warms up a bit
                    MSE_batch_linear_LOSS = self.loss_fn(fw_output_place_holder[est_FW_mask_loss],batch_target[FTT_FW_mask_loss])
                else:
                    MSE_batch_linear_LOSS = self.loss_fn(x_out_forward_batch[est_FW_mask_loss],batch_target[FTT_FW_mask_loss])
            else:
                for traj_id in range(len(traj_batch)):
                    non_zero_ids_in_FW = torch.cat((torch.tensor([[0]]), update_step_in_fw_mask[traj_id,0,:].nonzero()), dim=0) # Saves X_M0 and the hits where an update step was made
                    non_zero_ids_in_BW = x_out_batch[traj_id,0,:].nonzero() #filter out all the zeros
                    traj_batch[traj_id].x_estimated_FW = x_out_forward_batch[traj_id,:,non_zero_ids_in_FW]
                    traj_batch[traj_id].fw_gain = fw_gain[traj_id,:,non_zero_ids_in_FW]
                    traj_batch[traj_id].t_update_step_in_FW = update_step_in_fw_mask[traj_id,0,:].nonzero() #Saved for BW pass when we take saved FW pass
                    traj_batch[traj_id].x_estimated_BW = x_out_batch[traj_id,:,non_zero_ids_in_BW]
                    traj_batch[traj_id].bw_gain = bw_gain[traj_id,:,non_zero_ids_in_BW]
                    traj_batch[traj_id].bw_inov = bw_inov[traj_id,:,non_zero_ids_in_BW]
                    traj_batch[traj_id].bw_dx = bw_dx[traj_id,:,non_zero_ids_in_BW]
                if "HEAD" in self.SYSTEM_MODE.value:
                    # self.head_pipeline.model.set_requires_grad(False)
                    MSE_batch_linear_LOSS = self.head_pipeline.get_one_epoch_loss(traj_batch)  +  self.config.lambda_loss * self.loss_fn(x_out_batch[est_BW_mask_loss], batch_target[FTT_BW_mask_loss])#MSE Loss Function
                else:#BW w/o Head
                        if 'energy' in self.loss: #for init_energy mode only first time step velocities are saved , for energy mode all time step velocities are saved
                            x_out_velocities_reshaped = x_out_batch[est_BW_mask_loss].reshape(-1,3)
                            target_velocities_reshaped = batch_target[FTT_BW_mask_loss].reshape(-1,3)
                            x_out_energies = get_energy_from_velocities(x_out_velocities_reshaped[:,0],x_out_velocities_reshaped[:,1],x_out_velocities_reshaped[:,2])
                            target_energies = get_energy_from_velocities(target_velocities_reshaped[:,0],target_velocities_reshaped[:,1],target_velocities_reshaped[:,2])
                            MSE_batch_linear_LOSS = self.loss_fn(x_out_energies, target_energies)
                        else:
                            MSE_batch_linear_LOSS = self.loss_fn(x_out_batch[est_BW_mask_loss],batch_target[FTT_BW_mask_loss])

            return MSE_batch_linear_LOSS
    