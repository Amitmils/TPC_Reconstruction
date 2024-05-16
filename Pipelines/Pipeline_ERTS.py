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
from Tools.utils import get_mx_0,System_Mode,Trajectory_SS_Type,estimation_summary
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

        if self.config.use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def save(self):
        torch.save(self, self.PipelineName)

    def setssModel(self, ssModel):
        self.SysModel = ssModel

    def setModel(self, model):
        self.model = model
        self.model.config = self.config


    def setTrainingParams(self):
        self.batch_size = self.config.batch_size # Number of tracks in Batch
        self.total_num_phases = len(self.config.training_scheduler)

        self.current_phase = str(self.config.first_phase_id)
        self.num_epochs = np.sum([phase_config["n_epochs"] for phase_id,phase_config in enumerate(self.config.training_scheduler.values()) if phase_id>=self.config.first_phase_id])  # Number of Training Steps

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
        self.loss = self.config.training_scheduler[self.current_phase]["loss"] if "loss" in self.config.training_scheduler[self.current_phase] else 'all'
        self.weightDecay = self.config.wd # L2 Weight Regularization - Weight Decay
        # self.alpha = self.config.alpha # Composition loss factor
        # MSE LOSS Function
        self.loss_fn = nn.MSELoss(reduction='mean')
        self.set_optimizer()
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
        self.loss = self.config.training_scheduler[self.current_phase]["loss"] if "loss" in self.config.training_scheduler[self.current_phase] else 'all'
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
                # #save best model of Phase
                # torch.save(best_model, os.path.join(self.config.path_results,f"best-model-weights_P{self.current_phase}.pt"))
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
                    
                    torch.save(self.model.state_dict(), os.path.join(self.config.path_results,f"best-model-weights_P{self.current_phase}.pt"))
                    if int(self.current_phase) == self.total_num_phases-1:
                        torch.save(self.model.state_dict(), os.path.join(self.config.path_results,f"best-model-weights_FINAL.pt")) 

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

        estimation_summary(train_set,os.path.join("Simulations\Particle_Tracking\data",f"{os.path.basename(self.config.Dataset_path)}_LastTrainBatch_summary.csv"))
        estimation_summary(cv_set,os.path.join("Simulations\Particle_Tracking\data",f"{os.path.basename(self.config.Dataset_path)}_CV_summary.csv"))
        torch.save(best_model, os.path.join(self.config.path_results,f"best-model-weights_FINAL.pt"))
        # self.plot_training_summary() #TODO Why doesnt it work?
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


        if load_model_path is not None:
            print(f"Loading Model from path {load_model_path}")
            model_weights = torch.load(load_model_path, map_location=self.device) 
        else:
            print(f"Loading Model best-model-weights_FINAL")
            model_weights = torch.load(os.path.join(self.config.path_results,f'best-model-weights_FINAL.pt'), map_location=self.device) 
        # Set the loaded weights to the model
        self.model.load_state_dict(model_weights)

        # Test mode
        self.model.eval()
        # Init Hidden State
        self.model.init_hidden()
        torch.no_grad()
        with torch.no_grad():
            start = time.time()
            self.MSE_test_linear_avg = self.calculate_loss(test_set,SysModel,0,"Test")
            end = time.time()
            t = end - start

        # Average
        self.MSE_test_dB_avg = 10 * torch.log10(self.MSE_test_linear_avg)

        # Standard deviation
        # self.MSE_test_linear_std = torch.std(self.MSE_test_linear_arr, unbiased=True)

        # Confidence interval
        # self.test_std_dB = 10 * torch.log10(self.MSE_test_linear_std + self.MSE_test_linear_avg) - self.MSE_test_dB_avg

        # Print MSE and std
        str = self.modelName + "-" + "MSE Test:"
        print(str, self.MSE_test_dB_avg, "[dB]")
        # str = self.modelName + "-" + "STD Test:"
        # print(str, self.test_std_dB, "[dB]")
        # Print Run Time
        print("Inference Time:", t)
        estimation_summary(test_set,os.path.join("Simulations\Particle_Tracking\data",f"{os.path.basename(self.config.Dataset_path)}_Test_summary.csv"))

        return [self.MSE_test_linear_avg, self.MSE_test_dB_avg, t]

    def PlotTrain_RTS(self, MSE_KF_linear_arr, MSE_KF_dB_avg, MSE_RTS_linear_arr, MSE_RTS_dB_avg):
    
        self.Plot = Plot(self.folderName, self.modelName)

        self.Plot.NNPlot_epochs(self.train_set_size,self.num_epochs, self.test_set_size, MSE_KF_dB_avg, MSE_RTS_dB_avg,
                                self.MSE_test_dB_avg, self.MSE_cv_dB_epoch, self.MSE_train_dB_epoch)

        self.Plot.NNPlot_Hist(MSE_KF_linear_arr, MSE_RTS_linear_arr, self.MSE_test_linear_arr)

    def calculate_loss(self,traj_batch,SysModel,epoch_num,batch_type):
            
            self.model.batch_size = len(traj_batch)
            # Init Hidden State
            self.model.init_hidden()
            self.config.delta_t = torch.ones([self.model.batch_size]) * self.config.FTT_delta_t #forward pass has the finest Delta
    
            force_max_length = torch.inf if batch_type == "Test" or self.max_train_sequence_length == -1 else self.max_train_sequence_length #Used to limit BW loss
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
            x_out_batch = torch.zeros([len(traj_batch), SysModel.space_state_size, max_generated_traj_length_in_batch])
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
                clustered_in_generated_mask[ii,:,traj_batch[ii].t[1:]] = 1 #The first t is M1_0,
                m1_0_noisy,est_para = get_mx_0(traj_batch[ii].y.squeeze(-1))
                M1_0.append(m1_0_noisy.unsqueeze(0))
                ii += 1

            M1_0 = torch.cat(M1_0,dim=0)
            x_out_forward_batch[:, :, 0] = M1_0
            self.model.InitSequence(M1_0.unsqueeze(-1),max_clustered_traj_length_in_batch)

            # Forward Computation
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
                        next_ss_via_prediction_only[traj_id_in_batch_that_need_prediction] = self.model.f(x_out_forward_batch[traj_id_in_batch_that_need_prediction,:,fine_step_for_each_trajID_in_batch[traj_id_in_batch_that_need_prediction]],self.config.delta_t[traj_id_in_batch_that_need_prediction]).squeeze(-1)

                    # get distance of predictions to next observations
                    new_distances_to_obs[traj_id_in_batch_that_need_prediction] = torch.sqrt(torch.sum((next_ss_via_prediction_only[traj_id_in_batch_that_need_prediction,:3] - batch_y[traj_id_in_batch_that_need_prediction,:3,t])**2,dim=1))
                    # print(f"Est : {next_ss_via_prediction_only[traj_id_in_batch_that_need_prediction][0,:3]} , Obs {batch_y[traj_id_in_batch_that_need_prediction,:3,t]} , Dist {new_distances_to_obs}")
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
                    fine_step_for_each_trajID_in_batch[trajs_getting_closer]= torch.min(fine_step_for_each_trajID_in_batch[trajs_getting_closer] + 1,max_generated_traj_length_in_batch-1) #against overflow
                    distance_from_obs_for_each_trajID_in_batch[trajs_getting_closer] = new_distances_to_obs[trajs_getting_closer]

                    if any(time_stamp_to_update == max_generated_traj_length_in_batch):
                        print("Overflow in prediction!")

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
                relevant_ids_in_est_list = torch.zeros_like(updates_step_map) - 1
                est_BW_mask_loss = update_step_in_fw_mask.clone()
                FTT_BW_mask_loss = clustered_in_generated_mask.clone()

                for id_in_batch,x_out_FW in enumerate(x_out_forward_batch):
                    #for spoon feeding mode
                    updates_step_map[id_in_batch,:len(update_step_in_fw_mask[id_in_batch,0,:].nonzero())] = torch.flip(update_step_in_fw_mask[id_in_batch,0,:].nonzero(),dims=[0]).squeeze()
                    updates_step_map[id_in_batch,len(update_step_in_fw_mask[id_in_batch,0,:].nonzero())] = 0 #TODO combine both spoon feeding with normal and have only 1 mapping

                    #Velocity Loss Mask
                    if self.loss == 'velocity':
                        FTT_BW_mask_loss[id_in_batch] = False
                        FTT_BW_mask_loss[id_in_batch,-3:,traj_batch[id_in_batch].t[0]] = True 
                        est_BW_mask_loss[id_in_batch] = False
                        est_BW_mask_loss[id_in_batch,-3:,0] = True
                    elif self.loss == 'all':
                        # In BW, we update first point as well, add it to the loss mask
                        FTT_BW_mask_loss[id_in_batch,:,traj_batch[id_in_batch].t[0]] = True 
                        est_BW_mask_loss[id_in_batch,:,0] = True

                    #TODO organize is s.t. this will be in est_BW_mask_loss
                    relevant_ids_in_est = torch.arange(0,len_generated_hit_till_last_obs_hit[id_in_batch],self.config.delta_t_factor)
                    relevant_ids_in_FTT = torch.arange(traj_batch[id_in_batch].t[0],traj_batch[id_in_batch].t[-1],self.config.delta_t_factor)
                    relevant_ids_in_est_list[id_in_batch,-len(relevant_ids_in_est):] = relevant_ids_in_est
                    min_length = min(min(force_max_length,len(relevant_ids_in_FTT)),len(relevant_ids_in_est))
                    est_BW_mask[id_in_batch,:,relevant_ids_in_est[:min_length]] = 1 
                    FTT_BW_mask[id_in_batch,:,relevant_ids_in_FTT[:min_length]] = 1
                    generated_traj_length_mask[id_in_batch,:,len_generated_hit_till_last_obs_hit[id_in_batch]:] = 0 #nullify all hits after the hit thats closest to last observation

                if epoch_num < self.spoon_feeding:
                    # assert False,"Fix the delta_t"
                    x_out_batch[batch_ids,:,updates_step_map[:,0]] = x_out_forward_batch[batch_ids,:,updates_step_map[:,0]] #It was flipped such that 0 in the flipped is updates_step_map[:,0]
                    self.model.InitBackward(torch.unsqueeze(x_out_batch[batch_ids, :, updates_step_map[:,0]],2))
                    
                    self.config.delta_t =  (updates_step_map[:,0] - updates_step_map[:,1]) * self.config.FTT_delta_t
                    x_out_batch[batch_ids, :, updates_step_map[:,1]] = self.model(filter_x =x_out_forward_batch[batch_ids, :, updates_step_map[:,1]].unsqueeze(2),
                                                                                    filter_x_nexttime = x_out_batch[batch_ids, :, updates_step_map[:,0]].unsqueeze(2)).squeeze(2)
                    for k in range(2,updates_step_map.shape[1]):
                        end_of_traj = updates_step_map[:,k] == -1

                        if torch.all(end_of_traj):
                            break
                        self.config.delta_t =  (updates_step_map[:,k-1] - updates_step_map[:,k]) * self.config.FTT_delta_t
                        x_out_batch[~end_of_traj, :, updates_step_map[~end_of_traj,k]] = torch.squeeze(self.model(filter_x = torch.unsqueeze(x_out_forward_batch[batch_ids, :, updates_step_map[:,k]],2), 
                                                                                        filter_x_nexttime = torch.unsqueeze(x_out_batch[batch_ids, :, updates_step_map[:,k-1]],2),
                                                                                        smoother_x_tplus2 = torch.unsqueeze(x_out_batch[batch_ids, :, updates_step_map[:,k-2]],2)),2)[~end_of_traj,:]
                else:

                    self.config.delta_t = self.config.delta_t_factor * self.config.FTT_delta_t

                    x_out_batch[batch_ids,:,relevant_ids_in_est_list[:,-1]] = x_out_forward_batch[batch_ids,:,relevant_ids_in_est_list[:,-1]]
                    self.model.InitBackward(torch.unsqueeze(x_out_forward_batch[batch_ids,:,relevant_ids_in_est_list[:,-1]],2))
                    x_out_batch[batch_ids, :, relevant_ids_in_est_list[:,-2]] = torch.squeeze(self.model(filter_x = torch.unsqueeze(x_out_forward_batch[batch_ids, :, relevant_ids_in_est_list[:,-2]],2),
                                                                                    filter_x_nexttime = torch.unsqueeze(x_out_batch[batch_ids, :, relevant_ids_in_est_list[:,-1]],2)))

                    for k in range(3, 1 + max(torch.ceil(len_generated_hit_till_last_obs_hit/self.config.delta_t_factor).to(torch.int))):
                        end_of_traj = relevant_ids_in_est_list[:,-k] == -1

                        if torch.all(end_of_traj):
                            break
                        x_out_batch[~end_of_traj, :,relevant_ids_in_est_list[~end_of_traj,-k]] = self.model(filter_x = torch.unsqueeze(x_out_forward_batch[batch_ids, :, relevant_ids_in_est_list[:,-k]],2), 
                                                                                        filter_x_nexttime = torch.unsqueeze(x_out_batch[batch_ids, :, relevant_ids_in_est_list[:,-k + 1]],2),
                                                                                        smoother_x_tplus2 = torch.unsqueeze(x_out_batch[batch_ids, :, relevant_ids_in_est_list[:,-k + 2]],2))[~end_of_traj,:].squeeze(2)

            #Compute  loss
            if self.TRAINING_MODE == System_Mode.FW_ONLY:
                if epoch_num < self.spoon_feeding: #Till the KNET warms up a bit
                    MSE_batch_linear_LOSS = self.loss_fn(fw_output_place_holder[update_step_in_fw_mask],batch_target[clustered_in_generated_mask])
                else:
                    MSE_batch_linear_LOSS = self.loss_fn(x_out_forward_batch[update_step_in_fw_mask],batch_target[clustered_in_generated_mask])
            else:
                if epoch_num<self.spoon_feeding :
                    MSE_batch_linear_LOSS = self.loss_fn(x_out_batch[est_BW_mask_loss], batch_target[FTT_BW_mask_loss])
                else:
                    MSE_batch_linear_LOSS = self.loss_fn(x_out_batch[est_BW_mask], batch_target[FTT_BW_mask])

            for traj_id in range(len(traj_batch)):
                non_zero_ids_in_FW = update_step_in_fw_mask[traj_id,0,:].nonzero() #Only saves the ids which there was an update step
                non_zero_ids_in_BW = x_out_batch[traj_id,0,:].nonzero() #filter out all the zeros
                traj_batch[traj_id].x_estimated_FW = x_out_forward_batch[traj_id,:,non_zero_ids_in_FW].detach()
                traj_batch[traj_id].x_estimated_BW = x_out_batch[traj_id,:,non_zero_ids_in_BW].detach()


            # z = 0
            # print(f"gen : {get_mx_0(traj_batch[z].generated_traj.squeeze(-1))[1]}\n",
            #       f"y : {get_mx_0(traj_batch[z].y.squeeze(-1))[1]}\n",
            #       f"fw {get_mx_0(traj_batch[z].x_estimated_FW.squeeze(-1))[1]}\n",
            #       f"bw {get_mx_0(traj_batch[z].x_estimated_BW.squeeze(-1))[1]}\n"
            #       f" energy : {traj_batch[z].init_energy}")
            return MSE_batch_linear_LOSS
    