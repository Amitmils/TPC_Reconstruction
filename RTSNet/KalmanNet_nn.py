"""# **Class: KalmanNet**"""

import torch
import torch.nn as nn
import torch.nn.functional as func
import time

class KalmanNetNN(torch.nn.Module):

    ###################
    ### Constructor ###
    ###################
    def __init__(self):
        super().__init__()
    
    def NNBuild(self, SysModel, args):

        self.InitSystemDynamics(SysModel.f, SysModel.h, SysModel.m, SysModel.n)

        # Number of neurons in the 1st hidden layer
        #H1_KNet = (SysModel.m + SysModel.n) * (10) * 8

        # Number of neurons in the 2nd hidden layer
        #H2_KNet = (SysModel.m * SysModel.n) * 1 * (4)

        self.InitKGainNet(SysModel.prior_Q, SysModel.prior_Sigma, SysModel.prior_S, args)

    ######################################
    ### Initialize Kalman Gain Network ###
    ######################################
    def InitKGainNet(self, prior_Q, prior_Sigma, prior_S, system_config):

        self.seq_len_input = 1 # KNet calculates time-step by time-step
        self.batch_size = system_config.batch_size # Batch size

        self.prior_Q = prior_Q
        self.prior_Sigma = prior_Sigma
        self.prior_S = prior_S
        

        # GRU to track Q
        self.d_input_Q = self.space_state_size * system_config.input_dim_mult_KNet
        self.d_hidden_Q = self.space_state_size ** 2
        self.GRU_Q = nn.GRU(self.d_input_Q, self.d_hidden_Q)

        # GRU to track Sigma
        self.d_input_Sigma = self.d_hidden_Q + self.space_state_size * system_config.input_dim_mult_KNet
        self.d_hidden_Sigma = self.space_state_size ** 2
        self.GRU_Sigma = nn.GRU(self.d_input_Sigma, self.d_hidden_Sigma)
       
        # GRU to track S
        self.d_input_S = self.observation_vector_size ** 2 + 2 * self.observation_vector_size * system_config.input_dim_mult_KNet
        self.d_hidden_S = self.observation_vector_size ** 2
        self.GRU_S = nn.GRU(self.d_input_S, self.d_hidden_S)
        
        # Fully connected 1
        self.d_input_FC1 = self.d_hidden_Sigma
        self.d_output_FC1 = self.observation_vector_size ** 2
        self.FC1 = nn.Sequential(
                nn.Linear(self.d_input_FC1, self.d_output_FC1),
                nn.ReLU())

        # Fully connected 2
        self.d_input_FC2 = self.d_hidden_S + self.d_hidden_Sigma
        self.d_output_FC2 = self.observation_vector_size * self.space_state_size
        self.d_hidden_FC2 = self.d_input_FC2 * system_config.output_dim_mult_KNet
        self.FC2 = nn.Sequential(
                nn.Linear(self.d_input_FC2, self.d_hidden_FC2),
                nn.ReLU(),
                nn.Linear(self.d_hidden_FC2, self.d_output_FC2))

        # Fully connected 3
        self.d_input_FC3 = self.d_hidden_S + self.d_output_FC2
        self.d_output_FC3 = self.space_state_size ** 2
        self.FC3 = nn.Sequential(
                nn.Linear(self.d_input_FC3, self.d_output_FC3),
                nn.ReLU())

        # Fully connected 4
        self.d_input_FC4 = self.d_hidden_Sigma + self.d_output_FC3
        self.d_output_FC4 = self.d_hidden_Sigma
        self.FC4 = nn.Sequential(
                nn.Linear(self.d_input_FC4, self.d_output_FC4),
                nn.ReLU())
        
        # Fully connected 5
        self.d_input_FC5 = self.space_state_size
        self.d_output_FC5 = self.space_state_size * system_config.input_dim_mult_KNet
        self.FC5 = nn.Sequential(
                nn.Linear(self.d_input_FC5, self.d_output_FC5),
                nn.ReLU())

        # Fully connected 6
        self.d_input_FC6 = self.space_state_size
        self.d_output_FC6 = self.space_state_size * system_config.input_dim_mult_KNet
        self.FC6 = nn.Sequential(
                nn.Linear(self.d_input_FC6, self.d_output_FC6),
                nn.ReLU())
        
        # Fully connected 7
        self.d_input_FC7 = 2 * self.observation_vector_size
        self.d_output_FC7 = 2 * self.observation_vector_size * system_config.input_dim_mult_KNet
        self.FC7 = nn.Sequential(
                nn.Linear(self.d_input_FC7, self.d_output_FC7),
                nn.ReLU())

    ##################################
    ### Initialize System Dynamics ###
    ##################################
    def InitSystemDynamics(self, f, h, space_state_size, observation_vector_size):
        
        # Set State Evolution Function
        self.f = f
        self.space_state_size = space_state_size

        # Set Observation Function
        self.h = h
        self.observation_vector_size = observation_vector_size

    ###########################
    ### Initialize Sequence ###
    ###########################
    def InitSequence(self, M1_0, T=-1):
        """
        input M1_0 (torch.tensor): 1st moment of x at time 0 [batch_size, space_state_vector_size, 1]
        """
        self.T = T

        self.m1x_posterior = M1_0
        self.m1x_posterior_previous = self.m1x_posterior
        self.m1x_prior_previous = self.m1x_posterior
        self.y_previous = self.h(self.m1x_posterior)

    ######################
    ### Compute Priors ###
    ######################
    def step_prior(self,xt_minus_1 = None):
        # Predict the 1-st moment of x
        if xt_minus_1 is None:
            self.m1x_prior = self.f(self.m1x_posterior,self.config.FTT_delta_t)
        else:
            #in this case, our last posterior is not t-1 (data imputation mode)
            self.m1x_prior = self.f(xt_minus_1,self.config.FTT_delta_t)
            # print(f"t-1 : {xt_minus_1} , t : {self.m1x_prior}")


        # Predict the 1-st moment of y  
        self.m1y = self.h(self.m1x_prior)

    ##############################
    ### Kalman Gain Estimation ###
    ##############################
    def step_KGain_est(self, y):
        # both in size [batch_size, n]
        obs_diff = torch.squeeze(y,2) - torch.squeeze(self.y_previous,2) 
        obs_innov_diff = torch.squeeze(y,2) - torch.squeeze(self.m1y,2)
        # both in size [batch_size, m]
        fw_evol_diff = torch.squeeze(self.m1x_posterior,2) - torch.squeeze(self.m1x_posterior_previous,2)
        fw_update_diff = torch.squeeze(self.m1x_posterior,2) - torch.squeeze(self.m1x_prior_previous,2)

        obs_diff = func.normalize(obs_diff, p=2, dim=1, eps=1e-12, out=None)
        obs_innov_diff = func.normalize(obs_innov_diff, p=2, dim=1, eps=1e-12, out=None)
        fw_evol_diff = func.normalize(fw_evol_diff, p=2, dim=1, eps=1e-12, out=None)
        fw_update_diff = func.normalize(fw_update_diff, p=2, dim=1, eps=1e-12, out=None)

        # Kalman Gain Network Step
        KG = self.KGain_step(obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff)

        # Reshape Kalman Gain to a Matrix
        self.KGain = torch.reshape(KG, (self.batch_size, self.space_state_size, self.observation_vector_size))

    #######################
    ### Kalman Net Step ###
    #######################
    def KNet_step(self, y,xt_minus_1):

        # Compute Priors
        self.step_prior(xt_minus_1)

        # Compute Kalman Gain
        self.step_KGain_est(y)

        # Innovation
        dy = y - self.m1y # [batch_size, n, 1]

        # Compute the 1-st posterior moment
        INOV = torch.bmm(self.KGain, dy)
        self.m1x_posterior_previous = self.m1x_posterior
        self.m1x_posterior = self.m1x_prior + INOV

        #self.state_process_posterior_0 = self.state_process_prior_0
        self.m1x_prior_previous = self.m1x_prior

        # update y_prev
        self.y_previous = y

        # return
        return self.m1x_posterior

    ########################
    ### Kalman Gain Step ###
    ########################
    def KGain_step(self, obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff):

        def expand_dim(x):
            expanded = torch.empty(self.seq_len_input, self.batch_size, x.shape[-1])
            expanded[0, :, :] = x
            return expanded

        obs_diff = expand_dim(obs_diff)
        obs_innov_diff = expand_dim(obs_innov_diff)
        fw_evol_diff = expand_dim(fw_evol_diff)
        fw_update_diff = expand_dim(fw_update_diff)

        ####################
        ### Forward Flow ###
        ####################
        
        # FC 5
        in_FC5 = fw_evol_diff
        out_FC5 = self.FC5(in_FC5)

        # Q-GRU
        in_Q = out_FC5
        out_Q, self.h_Q = self.GRU_Q(in_Q, self.h_Q)

        # FC 6
        in_FC6 = fw_update_diff
        out_FC6 = self.FC6(in_FC6)

        # Sigma_GRU
        in_Sigma = torch.cat((out_Q, out_FC6), 2)
        out_Sigma, self.h_Sigma = self.GRU_Sigma(in_Sigma, self.h_Sigma)

        # FC 1
        in_FC1 = out_Sigma
        out_FC1 = self.FC1(in_FC1)

        # FC 7
        in_FC7 = torch.cat((obs_diff, obs_innov_diff), 2)
        out_FC7 = self.FC7(in_FC7)


        # S-GRU
        in_S = torch.cat((out_FC1, out_FC7), 2)
        out_S, self.h_S = self.GRU_S(in_S, self.h_S)


        # FC 2
        in_FC2 = torch.cat((out_Sigma, out_S), 2)
        out_FC2 = self.FC2(in_FC2)

        #####################
        ### Backward Flow ###
        #####################

        # FC 3
        in_FC3 = torch.cat((out_S, out_FC2), 2)
        out_FC3 = self.FC3(in_FC3)

        # FC 4
        in_FC4 = torch.cat((out_Sigma, out_FC3), 2)
        out_FC4 = self.FC4(in_FC4)

        # updating hidden state of the Sigma-GRU
        self.h_Sigma = out_FC4

        return out_FC2
    ###############
    ### Forward ###
    ###############
    def forward(self, y):
        return self.KNet_step(y)

    #########################
    ### Init Hidden State ###
    #########################
    def init_hidden_KNet(self):
        weight = next(self.parameters()).data
        hidden = weight.new(self.seq_len_input, self.batch_size, self.d_hidden_S).zero_()
        self.h_S = hidden.data
        self.h_S = self.prior_S.flatten().reshape(1, 1, -1).repeat(self.seq_len_input,self.batch_size, 1) # batch size expansion
        hidden = weight.new(self.seq_len_input, self.batch_size, self.d_hidden_Sigma).zero_()
        self.h_Sigma = hidden.data
        self.h_Sigma = self.prior_Sigma.flatten().reshape(1,1, -1).repeat(self.seq_len_input,self.batch_size, 1) # batch size expansion
        hidden = weight.new(self.seq_len_input, self.batch_size, self.d_hidden_Q).zero_()
        self.h_Q = hidden.data
        self.h_Q = self.prior_Q.flatten().reshape(1,1, -1).repeat(self.seq_len_input,self.batch_size, 1) # batch size expansion



