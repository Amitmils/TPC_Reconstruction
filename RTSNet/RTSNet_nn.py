"""# **Class: RTSNet**"""

import torch
import torch.nn as nn
import torch.nn.functional as func
import time

from RTSNet.KalmanNet_nn import KalmanNetNN

class RTSNetNN(KalmanNetNN):

    ###################
    ### Constructor ###
    ###################
    def __init__(self):
        super().__init__()

    #############
    ### Build ###
    #############
    def NNBuild(self, ssModel, system_config):

        self.InitSystemDynamics(ssModel.f, ssModel.h, ssModel.space_state_size, ssModel.observation_vector_size)

        self.InitKGainNet(ssModel.prior_Q, ssModel.prior_Sigma, ssModel.prior_S, system_config)

        self.InitRTSGainNet(ssModel.prior_Q, ssModel.prior_Sigma, system_config)

        self.KNET_params = []
        self.RTSNET_params = []
        for name,param in self.named_parameters():
            if 'bw' in name:
                self.RTSNET_params.append(param)
            else:
                self.KNET_params.append(param)


    #################################################
    ### Initialize Backward Smoother Gain Network ###
    #################################################
    def InitRTSGainNet(self, prior_Q, prior_Sigma, args):
        self.prior_Q = prior_Q
        self.prior_Sigma = prior_Sigma       


        # BW GRU to track Q
        self.d_input_Q_bw = self.space_state_size * args.input_dim_mult_RTSNet
        self.d_hidden_Q_bw = self.space_state_size ** 2
        self.GRU_Q_bw = nn.GRU(self.d_input_Q_bw, self.d_hidden_Q_bw)
    
        # BW GRU to track Sigma
        self.d_input_Sigma_bw = self.d_hidden_Q_bw + 2 * self.space_state_size * args.input_dim_mult_RTSNet
        self.d_hidden_Sigma_bw = self.space_state_size ** 2
        self.GRU_Sigma_bw = nn.GRU(self.d_input_Sigma_bw, self.d_hidden_Sigma_bw)
        
        # BW Fully connected 1
        self.d_input_FC1_bw = self.d_hidden_Sigma_bw # + self.d_hidden_Q
        self.d_output_FC1_bw = self.space_state_size * self.space_state_size
        self.d_hidden_FC1_bw = self.d_input_FC1_bw * args.output_dim_mult_RTSNet
        self.FC1_bw = nn.Sequential(
                nn.Linear(self.d_input_FC1_bw, self.d_hidden_FC1_bw),
                nn.ReLU(),
                nn.Linear(self.d_hidden_FC1_bw, self.d_output_FC1_bw))

        # BW Fully connected 2
        self.d_input_FC2_bw = self.d_hidden_Sigma_bw + self.d_output_FC1_bw
        self.d_output_FC2_bw = self.d_hidden_Sigma_bw
        self.FC2_bw = nn.Sequential(
                nn.Linear(self.d_input_FC2_bw, self.d_output_FC2_bw),
                nn.ReLU())
        
        # BW Fully connected 3
        self.d_input_FC3_bw = self.space_state_size
        self.d_output_FC3_bw = self.space_state_size * args.input_dim_mult_RTSNet
        self.FC3_bw = nn.Sequential(
                nn.Linear(self.d_input_FC3_bw, self.d_output_FC3_bw),
                nn.ReLU())

        # BW Fully connected 4
        self.d_input_FC4_bw = 2 * self.space_state_size
        self.d_output_FC4_bw = 2 * self.space_state_size * args.input_dim_mult_RTSNet
        self.FC4_bw = nn.Sequential(
                nn.Linear(self.d_input_FC4_bw, self.d_output_FC4_bw),
                nn.ReLU())

    ####################################
    ### Initialize Backward Sequence ###
    ####################################
    def InitBackward(self, filter_x):
        self.s_m1x_nexttime = filter_x
        self.dx = torch.ones_like(filter_x)

    ##############################
    ### Innovation Computation ###
    ##############################
    def S_Innovation(self, filter_x):
        self.filter_x_prior = self.f(filter_x.squeeze(-1),self.config.delta_t)
        # x_t+1|T - x_t+1|t (AMIT)
        self.dx = (self.s_m1x_nexttime - self.filter_x_prior) 

    ################################
    ### Smoother Gain Estimation ###
    ################################
    def step_RTSGain_est(self, filter_x_nexttime, smoother_x_tplus2):

        # Reshape and Normalize Delta tilde x_t+1 = x_t+1|T - x_t+1|t+1
        dm1x_tilde = self.s_m1x_nexttime - filter_x_nexttime
        dm1x_tilde_reshape = torch.squeeze(dm1x_tilde, 2) # size: (n_batch, m)
        bw_innov_diff = func.normalize(dm1x_tilde_reshape, p=2, dim=1, eps=1e-12, out=None)
        
        if smoother_x_tplus2 is None:
            # Reshape and Normalize Delta x_t+1 = x_t+1|t+1 - x_t+1|t (for t = T-1)
            dm1x_input2 = filter_x_nexttime - self.filter_x_prior
            dm1x_input2_reshape = torch.squeeze(dm1x_input2, 2) # size: (n_batch, m)
            bw_evol_diff = func.normalize(dm1x_input2_reshape, p=2, dim=1, eps=1e-12, out=None)
        else:
            # Reshape and Normalize Delta x_t+1|T = x_t+2|T - x_t+1|T (for t = 1:T-2)
            dm1x_input2 = smoother_x_tplus2 - self.s_m1x_nexttime
            dm1x_input2_reshape = torch.squeeze(dm1x_input2, 2) # size: (n_batch, m)
            bw_evol_diff = func.normalize(dm1x_input2_reshape, p=2, dim=1, eps=1e-12, out=None)

        # Feature 7:  x_t+1|T - x_t+1|t
        dm1x_f7 = self.s_m1x_nexttime - filter_x_nexttime
        dm1x_f7_reshape = torch.squeeze(dm1x_f7, 2) # size: (n_batch, m)
        bw_update_diff = func.normalize(dm1x_f7_reshape, p=2, dim=1, eps=1e-12, out=None)

        # Smoother Gain Network Step
        SG = self.RTSGain_step(bw_innov_diff, bw_evol_diff, bw_update_diff)

        # Reshape Smoother Gain to a Matrix
        self.SGain = torch.reshape(SG, (self.batch_size, self.space_state_size, self.space_state_size))

    ####################
    ### RTS Net Step ###
    ####################
    def RTSNet_step(self, filter_x, filter_x_nexttime, smoother_x_tplus2):
        # Compute Innovation
        self.S_Innovation(filter_x)

        # Compute Smoother Gain
        self.step_RTSGain_est(filter_x_nexttime, smoother_x_tplus2)

        # Compute the 1-st posterior moment
        INOV = torch.matmul(self.SGain, self.dx)
        self.s_m1x_nexttime = filter_x + INOV
        self.inov = INOV.detach()

        return self.s_m1x_nexttime

    ##########################
    ### Smoother Gain Step ###
    ##########################
    def RTSGain_step(self, bw_innov_diff, bw_evol_diff, bw_update_diff):

        def expand_dim(x):
            expanded = torch.empty(self.seq_len_input, self.batch_size, x.shape[-1])
            expanded[0, :, :] = x
            return expanded

        bw_innov_diff = expand_dim(bw_innov_diff)
        bw_evol_diff = expand_dim(bw_evol_diff)
        bw_update_diff = expand_dim(bw_update_diff)
        
        ####################
        ### Forward Flow ###
        ####################
        
        # FC 3
        in_FC3 = bw_update_diff
        out_FC3 = self.FC3_bw(in_FC3)

        # Q-GRU
        in_Q = out_FC3
        out_Q, self.h_Q_bw = self.GRU_Q_bw(in_Q, self.h_Q_bw)

        # FC 4
        in_FC4 = torch.cat((bw_innov_diff, bw_evol_diff), 2)
        out_FC4 = self.FC4_bw(in_FC4)

        # Sigma_GRU
        in_Sigma = torch.cat((out_Q, out_FC4), 2)
        out_Sigma, self.h_Sigma_bw = self.GRU_Sigma_bw(in_Sigma, self.h_Sigma_bw)

        # FC 1
        in_FC1 = out_Sigma
        out_FC1 = self.FC1_bw(in_FC1)

        #####################
        ### Backward Flow ###
        #####################

        # FC 2
        in_FC2 = torch.cat((out_Sigma, out_FC1), 2)
        out_FC2 = self.FC2_bw(in_FC2)

        # updating hidden state of the Sigma-GRU
        self.h_Sigma_bw = out_FC2

        self.feature_map = torch.cat((self.h_Sigma_bw.squeeze(0).detach(), self.h_Q_bw.squeeze(0).detach()), dim=1)
        return out_FC1

    ###############
    ### Forward ###
    ###############
    def forward(self, yt = None,xt_minus_1 = None, filter_x = None, filter_x_nexttime = None, smoother_x_tplus2 = None):
        if yt is None:
            # BW pass
            return self.RTSNet_step(filter_x, filter_x_nexttime, smoother_x_tplus2)
        else:
            # FW pass
            return self.KNet_step(yt,xt_minus_1)
    
    #########################
    ### Init Hidden State ###
    #########################
    def init_hidden(self):
        ### FW GRUs
        self.init_hidden_KNet()
        ### BW GRUs
        weight = next(self.parameters()).data
        hidden = weight.new(self.seq_len_input, self.batch_size, self.d_hidden_Q_bw).zero_()
        self.h_Q_bw = hidden.data
        self.h_Q_bw = self.prior_Q.flatten().reshape(1,1, -1).repeat(self.seq_len_input,self.batch_size, 1) # batch size expansion

        hidden = weight.new(self.seq_len_input, self.batch_size, self.d_hidden_Sigma_bw).zero_()
        self.h_Sigma_bw = hidden.data
        self.h_Sigma_bw = self.prior_Sigma.flatten().reshape(1,1, -1).repeat(self.seq_len_input,self.batch_size, 1) # batch size expansion

        
