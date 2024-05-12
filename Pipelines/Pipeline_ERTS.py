"""
This file contains the class Pipeline_ERTS, 
which is used to train and test RTSNet in both linear and non-linear cases.
"""

import torch
import torch.nn as nn
import time
import random
import numpy as np
from Plot import Plot_extended as Plot
from Tools.utils import get_mx_0,System_Mode,Trajectory_SS_Type
import os
import matplotlib.pyplot as plt
import time

class Pipeline_ERTS:

    def __init__(self, Time, folderName, modelName,system_config):
        super().__init__()
        self.config = system_config
        self.Time = Time
        self.folderName = folderName + '/'
        self.modelName = modelName
        self.modelFileName = self.folderName + "model_" + self.modelName + ".pt"
        self.PipelineName = self.folderName + "pipeline_" + self.modelName + ".pt"
        self.phase_change_epochs = [0] #for Visualization - aggregate the epochs which a phase changed

    def save(self):
        torch.save(self, self.PipelineName)

    def setssModel(self, ssModel):
        self.SysModel = ssModel

    def setModel(self, model):
        self.model = model
        self.model.config = self.config


    def setTrainingParams(self):
        if self.config.use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.num_epochs = np.sum([phase_config["n_epochs"] for phase_config in self.config.training_scheduler.values()])  # Number of Training Steps
        self.batch_size = self.config.batch_size # Number of Samples in Batch
        self.total_num_phases = len(self.config.training_scheduler)

        self.current_phase = str(self.config.first_phase_id)
        if self.current_phase !="0":
            prev_phase = str(int(self.current_phase)-1)
            weights_path = os.path.join(self.config.path_results,f"best-model-weights_P{prev_phase}.pt")
            model_weights = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(model_weights)
            print(f"Loaded Weights from {os.path.basename(weights_path)}")
        self.learningRate = self.config.training_scheduler[self.current_phase]["lr"] # Learning Rate of first phase
        self.num_epochs_in_phase = self.config.training_scheduler[self.current_phase]["n_epochs"]
        self.next_phase_change = self.num_epochs_in_phase # Which epoch to change to next phase
        self.max_train_sequence_length = self.config.training_scheduler[self.current_phase]["max_train_sequence_length"]
        self.TRAINING_MODE = System_Mode(self.config.training_scheduler[self.current_phase]["mode"])
        self.phase_modes = [self.TRAINING_MODE.value] #For visualization - set first mode type
        self.spoon_feeding = self.config.training_scheduler[self.current_phase]["spoon_feeding"] if "spoon_feeding" in self.config.training_scheduler[self.current_phase] else 0
        self.weightDecay = self.config.wd # L2 Weight Regularization - Weight Decay
        # self.alpha = self.config.alpha # Composition loss factor
        # MSE LOSS Function
        self.loss_fn = nn.MSELoss(reduction='mean')

        self.set_optimizer()
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min',factor=0.9, patience=20)
        if self.config.train == True:
            self.report_training_phase()

    def update_learning_rate(self,lr):
        self.learningRate = lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def set_optimizer(self):
        # Use the optim package to define an Optimizer that will update the weights of
        # the model for us. Here we will use Adam; the optim package contains many other
        # optimization algoriths. The first argument to the Adam constructor tells the
        # optimizer which Tensors it should update.

        if self.TRAINING_MODE == System_Mode.FW_BW:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learningRate, weight_decay=self.weightDecay)
        elif self.TRAINING_MODE == System_Mode.FW_ONLY:
            self.optimizer = torch.optim.Adam(self.model.KNET_params, lr=self.learningRate, weight_decay=self.weightDecay)
        elif self.TRAINING_MODE == System_Mode.BW_ONLY:
            self.optimizer = torch.optim.Adam(self.model.RTSNET_params, lr=self.learningRate, weight_decay=self.weightDecay)

    def update_training_mode(self):
        new_training_mode = self.config.training_scheduler[self.current_phase]["mode"]
        current_training_mode = self.TRAINING_MODE
        self.TRAINING_MODE = System_Mode(new_training_mode)
        if new_training_mode != current_training_mode:
            self.set_optimizer()

    def switch_to_next_phase(self):
        #Load best weights from phase
        weights_path = os.path.join(self.config.path_results,f"best-model-weights_P{self.current_phase}.pt")
        model_weights = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(model_weights)
        #continue to next phase
        self.current_phase = str(int(self.current_phase)+1)
        self.num_epochs_in_phase = self.config.training_scheduler[self.current_phase]["n_epochs"]
        self.spoon_feeding = self.config.training_scheduler[self.current_phase]["spoon_feeding"] if "spoon_feeding" in self.config.training_scheduler[self.current_phase] else 0
        self.next_phase_change = self.next_phase_change + self.num_epochs_in_phase
        self.max_train_sequence_length = self.config.training_scheduler[self.current_phase]["max_train_sequence_length"]
        self.update_learning_rate(self.config.training_scheduler[self.current_phase]["lr"])
        self.update_training_mode()
        self.report_training_phase()


    def report_training_phase(self):
        print(f"\n######## Entered Training Phase {self.current_phase} ########\n\n"
              f"Num. of epochs in phase : {self.num_epochs_in_phase}\n"
              f"Next Switch Epoch : {self.next_phase_change}\n"
              f"Learning Rate : {self.learningRate}\n"
              f"Max Seq Length : {self.max_train_sequence_length}\n"
              f"Mode : {self.TRAINING_MODE.value}\n"
              f"Spoon Feeding: {self.spoon_feeding > 0}\n"
              f"Training {sum(p.numel() for p in self.optimizer.param_groups[0]['params'])} parameters\n\n"
              "########################################\n")

    def NNTrain(self, SysModel, train_set, cv_set):

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

        for ti in range(0, self.num_epochs):
            start = time.time()
            if ti == self.next_phase_change:
                #save best model of Phase
                torch.save(best_model, os.path.join(self.config.path_results,f"best-model-weights_P{self.current_phase}.pt"))
                self.MSE_cv_opt_dB_phase[int(self.current_phase)] = self.MSE_cv_dB_opt
                self.MSE_cv_opt_id_phase[int(self.current_phase)] = self.MSE_cv_idx_opt
                self.MSE_cv_dB_opt = 1000
                self.phase_change_epochs.append(ti) #for Visualization - aggregate the epochs which a phase changed
                self.switch_to_next_phase()
                self.phase_modes.append(self.TRAINING_MODE.value)#for Visualization

            ###############################
            ### Training Sequence Batch ###
            ###############################

            # Training Mode
            self.model.train()
            self.optimizer.zero_grad()

            # Randomly select N_B training sequencesgit stat
            assert self.batch_size <= self.train_set_size # N_B must be smaller than N_E
            n_e = self.config.force_batch if len(self.config.force_batch) else random.sample(range(self.train_set_size), k=self.batch_size)
            train_batch = [train_set[idx] for idx in n_e if train_set[idx].traj_length>-1]

            MSE_train_batch_linear_LOSS = self.calculate_loss(train_batch,SysModel,ti,"Train")

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
            MSE_train_batch_linear_LOSS.backward(retain_graph=True)

            # Calling the step function on an Optimizer makes an update to its
            # parameters
            self.optimizer.step()

            # self.scheduler.step(self.MSE_cv_dB_epoch[ti])

            #################################
            ### Validation Sequence Batch ###
            #################################

            # Cross Validation Mode
            self.model.eval()

            with torch.no_grad():
                MSE_cv_linear_LOSS = self.calculate_loss(cv_set,SysModel,ti,"CV")

                # dB Loss
                self.MSE_cv_linear_epoch[ti] = MSE_cv_linear_LOSS.item()
                self.MSE_cv_dB_epoch[ti] = 10 * torch.log10(self.MSE_cv_linear_epoch[ti])

                if (self.MSE_cv_dB_epoch[ti] < self.MSE_cv_dB_opt or ti==0):
                    self.MSE_cv_dB_opt = self.MSE_cv_dB_epoch[ti]
                    self.MSE_cv_idx_opt = ti
                    best_model = self.model.state_dict().copy()
                    
                    # torch.save(self.model.state_dict(), os.path.join(self.config.path_results,f"best-model-weights_P{self.current_phase}.pt"))
                    # if int(self.current_phase) == self.total_num_phases-1:
                    #     torch.save(self.model.state_dict(), os.path.join(self.config.path_results,f"best-model-weights_FINAL.pt")) 

            # cv_set[0].x_estimated_BW = x_out_cv[0].clone().detach()
            # cv_set[0].x_estimated_FW = x_out_cv_forward[0].clone().detach()
            # cv_set[0].traj_plots([Trajectory_SS_Type.Estimated_FW,Trajectory_SS_Type.GTT,Trajectory_SS_Type.OT])

            ########################
            ### Training Summary ###
            ########################
            print(f"Time : {time.time()-start}")
            print(f"P{self.current_phase}",ti, "MSE Training :", self.MSE_train_dB_epoch[ti], "[dB]", "MSE Validation :", self.MSE_cv_dB_epoch[ti],
                  "[dB]")

            if (ti > 0):
                d_train = self.MSE_train_dB_epoch[ti] - self.MSE_train_dB_epoch[ti - 1]
                d_cv = self.MSE_cv_dB_epoch[ti] - self.MSE_cv_dB_epoch[ti - 1]
                print("diff MSE Training :", d_train, "[dB]", "diff MSE Validation :", d_cv, "[dB]")

            print("Optimal idx:", self.MSE_cv_idx_opt, "Optimal :", self.MSE_cv_dB_opt, "[dB]")

        torch.save(best_model, os.path.join(self.config.path_results,f"best-model-weights_FINAL.pt"))
        self.plot_training_summary()
        return [self.MSE_cv_linear_epoch, self.MSE_cv_dB_epoch, self.MSE_train_linear_epoch, self.MSE_train_dB_epoch]

    def plot_training_summary(self):
        self.MSE_cv_opt_dB_phase[int(self.current_phase)] = self.MSE_cv_dB_opt
        self.MSE_cv_opt_id_phase[int(self.current_phase)] = self.MSE_cv_idx_opt

        plt.figure()
        plt.plot(range(len(self.MSE_train_dB_epoch)),self.MSE_train_dB_epoch,label='Train loss',linewidth=0.5)
        plt.plot(range(len(self.MSE_cv_dB_epoch)),self.MSE_cv_dB_epoch,label='CV loss',linewidth=0.8,linestyle='--')
        plt.scatter(self.MSE_cv_opt_id_phase[:int(self.current_phase) + 1], self.MSE_cv_opt_dB_phase[:int(self.current_phase)+1], color='green', marker='o',s=10,label='Opt MSE')  # Mark specific points
        # Mark best MSE in each phase
        # for i in range(int(self.current_phase)+1):
        #     plt.text((self.phase_change_epochs[i] + self.phase_change_epochs[i+1])/2 - 10, plt.gca().get_ylim()[1] + 0.5, f'Opt MSE {round(self.MSE_cv_opt_dB_phase[i].item(),2)})', fontsize=10)
        # Mark Phase switches
        for epoch in self.phase_change_epochs:
            plt.axvline(x=epoch, color='red', linestyle='--',linewidth=0.7)
        for phase_id,epoch in enumerate(self.phase_change_epochs):
            plt.text(epoch, plt.gca().get_ylim()[1] + 0.7, f'P{phase_id}\n ({self.phase_modes[phase_id]})', ha='center')

        plt.xlabel("Epoch")
        plt.ylabel('MSE [dB]')
        plt.title("Training Losses", y=1.09)
        plt.legend()
        plt.show()

    def NNTest(self, SysModel, test_set,load_model_path=None):
        
        self.TEST_MODE = System_Mode(self.config.test_mode)
        self.test_set_size = self.model.batch_size = len(test_set)

        traj_lengths_in_test = torch.tensor([traj.traj_length for traj in test_set])
        max_traj_length_in_test = torch.max(traj_lengths_in_test)

        x_out_test_forward         = torch.zeros([self.test_set_size, SysModel.space_state_size, max_traj_length_in_test])
        x_out_test_forward_flipped = torch.zeros([self.test_set_size, SysModel.space_state_size, max_traj_length_in_test])
        x_out_test                 = torch.zeros([self.test_set_size, SysModel.space_state_size,max_traj_length_in_test])
        x_out_test_flipped         = torch.zeros([self.test_set_size, SysModel.space_state_size,max_traj_length_in_test])
        test_target                = torch.zeros([self.test_set_size, SysModel.space_state_size,max_traj_length_in_test])
        y_test                     = torch.zeros([self.test_set_size, SysModel.observation_vector_size,max_traj_length_in_test])
        mask_test                  = torch.zeros([self.test_set_size, SysModel.space_state_size, max_traj_length_in_test])
        
        self.MSE_test_linear_arr = torch.zeros([self.test_set_size])

        self.MSE_test_linear_arr = torch.zeros([self.test_set_size])

        # MSE LOSS Function
        loss_fn = nn.MSELoss(reduction='mean')

        if load_model_path is not None:
            print(f"Loading Model from path {load_model_path}")
            model_weights = torch.load(load_model_path, map_location=self.device) 
        else:
            model_weights = torch.load(os.path.join(self.config.path_results,f'best-model-weights_FINAL.pt'), map_location=self.device) 
        # Set the loaded weights to the model
        self.model.load_state_dict(model_weights)

        # Test mode
        self.model.eval()
        # Init Hidden State
        self.model.init_hidden()
        torch.no_grad()

        start = time.time()
        M1_0 = []
        for ii,traj in enumerate(test_set):
            y_test[ii,:,:traj.traj_length]      = traj.y[:,:traj.traj_length]
            test_target[ii,:,:traj.traj_length] = traj.x_real[:,:traj.traj_length]
            mask_test[ii,:,:traj.traj_length]   = 1
            m1_0_real,para_real = get_mx_0(test_set[ii].x_real)
            m1_0_noisy,para_noisy= get_mx_0(test_set[ii].y,forced_phi=para_real['initial_phi'])
            M1_0.append(m1_0_noisy.unsqueeze(0))

        M1_0 = torch.cat(M1_0,dim=0)
        x_out_test_forward[:, :, 0] = M1_0
        self.model.InitSequence(M1_0.unsqueeze(-1),max_traj_length_in_test)


        # Forward Computation
        for t in range(0, max_traj_length_in_test):
            x_out_test_forward[:, :, t] = torch.squeeze(self.model(yt = torch.unsqueeze(y_test[:, :, t],2)))

        if self.TEST_MODE !=System_Mode.FW_ONLY:
            # Flip the order (this is needed because each trajectoy has a different length)
            for id,x_out_FW in enumerate(x_out_test_forward):
                x_out_test_forward_flipped[id,:,:traj_lengths_in_test[id]] = torch.flip(x_out_FW[:,:traj_lengths_in_test[id]],dims=[1])

            x_out_test_flipped[:,:,0] = x_out_test_forward_flipped[:,:,0]
            self.model.InitBackward(torch.unsqueeze(x_out_test_flipped[:, :, 0],2))
            x_out_test_flipped[:, :, 1] = torch.squeeze(self.model(filter_x = torch.unsqueeze(x_out_test_forward_flipped[:, :, 1],2),
                                                                filter_x_nexttime = torch.unsqueeze(x_out_test_forward_flipped[:, :, 0],2)))  

            # Backward Computation; index k happens after k+1. 
            # E.g., if the trajectory is of length 100. time stamp 100 is at k=0 , time stamp 99 is at k=1. k = 100 - t
            for k in range(2,max_traj_length_in_test):
                x_out_test_flipped[:, :, k] = torch.squeeze(self.model(filter_x = torch.unsqueeze(x_out_test_forward_flipped[:, :, k],2), 
                                                                    filter_x_nexttime = torch.unsqueeze(x_out_test_forward_flipped[:, :, k-1],2),
                                                                    smoother_x_tplus2 = torch.unsqueeze(x_out_test_flipped[:, :, k-2],2)))

            # Flip back to original order
            for id,x_out_flipped in enumerate(x_out_test_flipped):
                x_out_test[id,:,:traj_lengths_in_test[id]] = torch.flip(x_out_flipped[:,:traj_lengths_in_test[id]],dims=[1])

        test_set[0].x_estimated_FW = x_out_test_forward[0,:,:].detach()
        test_set[0].x_estimated_BW = x_out_test[0,:,:].detach()
        test_set[0].traj_plots([Trajectory_SS_Type.GTT,Trajectory_SS_Type.OT,Trajectory_SS_Type.Estimated_FW])

        end = time.time()
        t = end - start

        # MSE loss
        for j in range(self.test_set_size ):
            if self.TRAINING_MODE ==System_Mode.FW_ONLY:
                self.MSE_test_linear_arr[j] = loss_fn(x_out_test_forward[j,:,:] * mask_test[j,:,:], test_target[j,:,:]).item()
            else:
                self.MSE_test_linear_arr[j] = loss_fn(x_out_test[j,:,:] * mask_test[j,:,:], test_target[j,:,:]).item()
        
        # Average
        self.MSE_test_linear_avg = torch.mean(self.MSE_test_linear_arr)
        self.MSE_test_dB_avg = 10 * torch.log10(self.MSE_test_linear_avg)

        # Standard deviation
        self.MSE_test_linear_std = torch.std(self.MSE_test_linear_arr, unbiased=True)

        # Confidence interval
        self.test_std_dB = 10 * torch.log10(self.MSE_test_linear_std + self.MSE_test_linear_avg) - self.MSE_test_dB_avg

        # Print MSE and std
        str = self.modelName + "-" + "MSE Test:"
        print(str, self.MSE_test_dB_avg, "[dB]")
        str = self.modelName + "-" + "STD Test:"
        print(str, self.test_std_dB, "[dB]")
        # Print Run Time
        print("Inference Time:", t)

        return [self.MSE_test_linear_arr, self.MSE_test_linear_avg, self.MSE_test_dB_avg, x_out_test, t]

    def PlotTrain_RTS(self, MSE_KF_linear_arr, MSE_KF_dB_avg, MSE_RTS_linear_arr, MSE_RTS_dB_avg):
    
        self.Plot = Plot(self.folderName, self.modelName)

        self.Plot.NNPlot_epochs(self.train_set_size,self.num_epochs, self.test_set_size, MSE_KF_dB_avg, MSE_RTS_dB_avg,
                                self.MSE_test_dB_avg, self.MSE_cv_dB_epoch, self.MSE_train_dB_epoch)

        self.Plot.NNPlot_Hist(MSE_KF_linear_arr, MSE_RTS_linear_arr, self.MSE_test_linear_arr)

    def calculate_loss(self,traj_batch,SysModel,epoch_num,batch_type):
            
            self.model.batch_size = len(traj_batch)
            # Init Hidden State
            self.model.init_hidden()
    
            force_max_length = torch.inf if self.max_train_sequence_length == -1 else self.max_train_sequence_length #Used to limit BW loss
            clustered_traj_lengths_in_batch = torch.tensor([traj.x_real.shape[1] for traj in traj_batch])
            generated_traj_lengths_in_batch = torch.tensor([traj.generated_traj.shape[1] for traj in traj_batch])
            max_clustered_traj_length_in_batch = torch.max(clustered_traj_lengths_in_batch)
            max_generated_traj_length_in_batch = torch.max(generated_traj_lengths_in_batch)

            ## Init Training Batch tensors##
            #lengths before imputation
            batch_y = torch.zeros([len(traj_batch), SysModel.observation_vector_size, max_clustered_traj_length_in_batch])

            #lengths after imputation
            batch_target = torch.zeros([len(traj_batch), SysModel.space_state_size, max_generated_traj_length_in_batch])
            fw_output_place_holder = torch.zeros([len(traj_batch), SysModel.space_state_size, max_generated_traj_length_in_batch])
            x_out_forward_batch = torch.zeros([len(traj_batch), SysModel.space_state_size, max_generated_traj_length_in_batch])
            x_out_forward_batch_flipped = torch.zeros([len(traj_batch), SysModel.space_state_size, max_generated_traj_length_in_batch])
            x_out_batch = torch.zeros([len(traj_batch), SysModel.space_state_size, max_generated_traj_length_in_batch])
            x_out_batch_flipped = torch.zeros([len(traj_batch), SysModel.space_state_size, max_generated_traj_length_in_batch])
            est_BW_mask = torch.zeros([len(traj_batch), SysModel.space_state_size, max_generated_traj_length_in_batch],dtype=torch.bool)
            FTT_BW_mask = torch.zeros([len(traj_batch), SysModel.space_state_size, max_generated_traj_length_in_batch],dtype=torch.bool)
            generated_traj_length_mask = torch.zeros([len(traj_batch), SysModel.space_state_size, max_generated_traj_length_in_batch],dtype=torch.bool)#zeroify values that are after the traj length
            clustered_in_generated_mask = torch.zeros([len(traj_batch), SysModel.space_state_size, max_generated_traj_length_in_batch],dtype=torch.bool)#if index is True , that SS is GTT and FTT -- for FW Pass Loss
            update_step_in_fw_mask = torch.zeros([len(traj_batch), SysModel.space_state_size, max_generated_traj_length_in_batch],dtype=torch.bool)# Mark which time steps in FW had update step from KF -- for FW Pass Loss
            updates_step_map = torch.zeros([len(traj_batch), max_generated_traj_length_in_batch],dtype=torch.int) + -1# For BW



            M1_0 = []
            for ii in range(len(traj_batch)):
                batch_y[ii,:,:clustered_traj_lengths_in_batch[ii]] = traj_batch[ii].y[:3,:clustered_traj_lengths_in_batch[ii]].squeeze(-1)
                batch_target[ii,:,:generated_traj_lengths_in_batch[ii]] = traj_batch[ii].generated_traj[:,:generated_traj_lengths_in_batch[ii]].squeeze(-1)
                generated_traj_length_mask[ii,:,:generated_traj_lengths_in_batch[ii]] = 1
                clustered_in_generated_mask[ii,:,traj_batch[ii].t[1:]] = 1 #The first t is M1_0, normalize such that its 0 and remove since we dont estimate w/ KNET
                m1_0_noisy,est_para = get_mx_0(traj_batch[ii].y.squeeze(-1))
                M1_0.append(m1_0_noisy.unsqueeze(0))
                ii += 1

            M1_0 = torch.cat(M1_0,dim=0)
            x_out_forward_batch[:, :, 0] = M1_0
            self.model.InitSequence(M1_0.unsqueeze(-1),max_clustered_traj_length_in_batch)

            # Forward Computation
            self.config.FTT_delta_t = traj_batch[0].delta_t #TODO rewrite 
            fine_step_for_each_trajID_in_batch = torch.zeros(len(traj_batch),dtype=torch.int)
            for t in range(1,max_clustered_traj_length_in_batch):
                if not(t%10):
                    print(f" {batch_type} t = {t}")
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
                        next_ss_via_prediction_only[traj_id_in_batch_that_need_prediction] = self.model.f(x_out_forward_batch[traj_id_in_batch_that_need_prediction,:,fine_step_for_each_trajID_in_batch[traj_id_in_batch_that_need_prediction]],self.config.FTT_delta_t).squeeze(-1)

                    # get distance of predictions to next observations
                    new_distances_to_obs[traj_id_in_batch_that_need_prediction] = torch.sqrt(torch.sum((next_ss_via_prediction_only[traj_id_in_batch_that_need_prediction,:3] - batch_y[traj_id_in_batch_that_need_prediction,:3,t])**2,dim=1))

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
                    x_out_forward_batch[trajs_getting_closer,:,fine_step_for_each_trajID_in_batch[trajs_getting_closer]+1] = next_ss_via_prediction_only[trajs_getting_closer]
                    fine_step_for_each_trajID_in_batch[trajs_getting_closer]+=1
                    distance_from_obs_for_each_trajID_in_batch[trajs_getting_closer] = new_distances_to_obs[trajs_getting_closer]

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
                if self.TRAINING_MODE == System_Mode.FW_ONLY and epoch_num < self.spoon_feeding: #Till the KNET warms up a bit
                    fw_output_place_holder[~traj_id_in_batch_finished,:,fine_step_for_each_trajID_in_batch[~traj_id_in_batch_finished]] = output[~traj_id_in_batch_finished]
                else:
                    x_out_forward_batch[~traj_id_in_batch_finished,:,fine_step_for_each_trajID_in_batch[~traj_id_in_batch_finished]] = output[~traj_id_in_batch_finished]

            # Backward Computation
            if self.TRAINING_MODE != System_Mode.FW_ONLY:
                batch_ids = torch.arange(len(traj_batch))
                len_generated_hit_till_last_obs_hit = fine_step_for_each_trajID_in_batch + 1 #For every traj, closest gen hit id for the LAST obs hit
                # Flip the order (this is needed because each trajectoy has a different length)
                for id_in_batch,x_out_FW in enumerate(x_out_forward_batch):
                    updates_step_map[id_in_batch,:len(update_step_in_fw_mask[id_in_batch,0,:].nonzero())] = torch.flip(update_step_in_fw_mask[id_in_batch,0,:].nonzero(),dims=[0]).squeeze()
                    updates_step_map[id_in_batch,len(update_step_in_fw_mask[id_in_batch,0,:].nonzero())] = 0
                    min_length = min(min(force_max_length,len_generated_hit_till_last_obs_hit[id_in_batch]),traj_batch[id_in_batch].t[-1]-traj_batch[id_in_batch].t[0])
                    est_BW_mask[id_in_batch,:,:min_length] = 1
                    FTT_BW_mask[id_in_batch,:,traj_batch[id_in_batch].t[0]:traj_batch[id_in_batch].t[0] + min_length] = 1
                    generated_traj_length_mask[id_in_batch,:,len_generated_hit_till_last_obs_hit[id_in_batch]:] = 0 #nullify all hits after the hit thats closest to last observation
                    x_out_forward_batch_flipped[id_in_batch,:,:len_generated_hit_till_last_obs_hit[id_in_batch]] = torch.flip(x_out_FW[:,:len_generated_hit_till_last_obs_hit[id_in_batch]],dims=[1])

                if epoch_num < self.spoon_feeding:
                    self.config.FTT_delta_t = traj_batch[0].delta_t * 20 #TODO rewrite 
                    x_out_batch[batch_ids,:,updates_step_map[:,0]] = x_out_forward_batch[batch_ids,:,updates_step_map[:,0]] #It was flipped such that 0 in the flipped is updates_step_map[:,0]
                    self.model.InitBackward(torch.unsqueeze(x_out_batch[batch_ids, :, updates_step_map[:,0]],2))
                    x_out_batch[batch_ids, :, updates_step_map[:,1]] = self.model(filter_x =x_out_forward_batch[batch_ids, :, updates_step_map[:,1]].unsqueeze(2),
                                                                                    filter_x_nexttime = x_out_batch[batch_ids, :, updates_step_map[:,0]].unsqueeze(2)).squeeze(2)
                    # print(f"1 : dx : {torch.norm(self.model.dx)} , gain {torch.norm(self.model.SGain)}")

                    for k in range(2,updates_step_map.shape[1]):
                        end_of_traj = updates_step_map[:,k] == -1

                        if torch.all(end_of_traj):
                            break
                        x_out_batch[~end_of_traj, :, updates_step_map[~end_of_traj,k]] = torch.squeeze(self.model(filter_x = torch.unsqueeze(x_out_forward_batch[batch_ids, :, updates_step_map[:,k]],2), 
                                                                                        filter_x_nexttime = torch.unsqueeze(x_out_batch[batch_ids, :, updates_step_map[:,k-1]],2),
                                                                                        smoother_x_tplus2 = torch.unsqueeze(x_out_batch[batch_ids, :, updates_step_map[:,k-2]],2)),2)[~end_of_traj,:]
                        # print(f"{k} : dx : {torch.norm(self.model.dx)} , gain {torch.norm(self.model.SGain)}")
                else:
                    #TODO Remove Flipped - Not Efficient
                    x_out_batch_flipped[:,:,0] = x_out_forward_batch_flipped[:,:,0]
                    self.model.InitBackward(torch.unsqueeze(x_out_batch_flipped[:, :, 0],2))
                    x_out_batch_flipped[:, :, 1] = torch.squeeze(self.model(filter_x = torch.unsqueeze(x_out_forward_batch_flipped[:, :, 1],2),
                                                                                    filter_x_nexttime = torch.unsqueeze(x_out_batch_flipped[:, :, 0],2)))
                    dx_vector = torch.zeros([self.batch_size,x_out_batch_flipped.shape[2]])
                    dx_vector[:,1] = torch.norm(self.model.dx)
                    # print(f"1 : dx : {torch.norm(self.model.dx)} , gain {torch.norm(self.model.SGain)}")
                    #  index k happens after k+1. 
                    # E.g., if the trajectory is of length 100. time stamp 100 is at k=0 , time stamp 99 is at k=1. k = 100 - t
                    for k in range(2,max(len_generated_hit_till_last_obs_hit)):
                        x_out_batch_flipped[:, :, k] = torch.squeeze(self.model(filter_x = torch.unsqueeze(x_out_forward_batch_flipped[:, :, k],2), 
                                                                                        filter_x_nexttime = torch.unsqueeze(x_out_batch_flipped[:, :, k - 1],2),
                                                                                        smoother_x_tplus2 = torch.unsqueeze(x_out_batch_flipped[:, :, k-2],2)))
                        dx_vector[:,k] = torch.norm(self.model.dx)                                                
                        # print(f"{k} : dx : {torch.norm(self.model.dx)} , gain {torch.norm(self.model.SGain)}")
                        
                    # Flip back to original order
                    for id_in_batch,x_out_flipped in enumerate(x_out_batch_flipped):
                        x_out_batch[id_in_batch,:,:len_generated_hit_till_last_obs_hit[id_in_batch]] = torch.flip(x_out_flipped[:,:len_generated_hit_till_last_obs_hit[id_in_batch]],dims=[1])

            traj_batch[0].x_estimated_BW = x_out_batch[0].clone().detach()
            traj_batch[0].x_estimated_FW = x_out_forward_batch[0].clone().detach()
            # train_batch[0].traj_plots([Trajectory_SS_Type.Estimated_BW,Trajectory_SS_Type.Estimated_FW,Trajectory_SS_Type.Real])

            #Compute train loss
            if self.TRAINING_MODE == System_Mode.FW_ONLY:
                if epoch_num < self.spoon_feeding: #Till the KNET warms up a bit
                    MSE_batch_linear_LOSS = self.loss_fn(fw_output_place_holder[update_step_in_fw_mask],batch_target[clustered_in_generated_mask])
                else:
                    MSE_batch_linear_LOSS = self.loss_fn(x_out_forward_batch[update_step_in_fw_mask],batch_target[clustered_in_generated_mask])
            else:
                if epoch_num<self.spoon_feeding :
                    MSE_batch_linear_LOSS = self.loss_fn(x_out_batch[update_step_in_fw_mask], batch_target[clustered_in_generated_mask])
                else:
                    MSE_batch_linear_LOSS = self.loss_fn(x_out_batch[est_BW_mask], batch_target[FTT_BW_mask])

                for i in range(self.batch_size):
                    if epoch_num < self.spoon_feeding:
                        print(self.loss_fn(x_out_batch[i,update_step_in_fw_mask[i]],batch_target[i,clustered_in_generated_mask[i]]))
                    else:
                        print(self.loss_fn(x_out_batch[i,est_BW_mask[i]],batch_target[i,FTT_BW_mask[i]]))

            return MSE_batch_linear_LOSS
    