import torch
import torch.nn as nn
import time
from EKF import ExtendedKalmanFilter
from Extended_RTS_Smoother import Extended_rts_smoother
from Extended_data import N_T

def S_Test(SysModel, test_input, test_target, modelKnowledge = 'full', randomInit = False):

    # LOSS
    loss_rts = nn.MSELoss(reduction='mean')

    # MSE [Linear]
    MSE_ERTS_linear_arr = torch.empty(N_T)
    start = time.time()
    EKF = ExtendedKalmanFilter(SysModel, modelKnowledge)  
    ERTS = Extended_rts_smoother(SysModel, modelKnowledge)
    # Allocate empty list for output
    ERTS_out = []
    j=0
    
    for sequence_target,sequence_input in zip(test_target,test_input):
        if(randomInit):
            EKF.InitSequence(torch.unsqueeze(SysModel.m1x_0_rand[j,:],1), SysModel.m2x_0)   
        else:
            EKF.InitSequence(SysModel.m1x_0, SysModel.m2x_0)

        EKF.GenerateSequence(sequence_input, sequence_input.size()[-1])
        ERTS.GenerateSequence(EKF.x, EKF.sigma, sequence_input.size()[-1])
        MSE_ERTS_linear_arr[j] = loss_rts(ERTS.s_x, sequence_target).item()
        ERTS_out.append(ERTS.s_x)
        j=j+1
    end = time.time()
    t = end - start
    MSE_ERTS_linear_avg = torch.mean(MSE_ERTS_linear_arr)
    MSE_ERTS_dB_avg = 10 * torch.log10(MSE_ERTS_linear_avg)

    # Standard deviation
    MSE_ERTS_linear_std = torch.std(MSE_ERTS_linear_arr, unbiased=True)

    # Confidence interval
    ERTS_std_dB = 10 * torch.log10(MSE_ERTS_linear_std + MSE_ERTS_linear_avg) - MSE_ERTS_dB_avg

    print("Extended RTS Smoother - MSE LOSS:", MSE_ERTS_dB_avg, "[dB]")
    print("Extended RTS Smoother - STD:", ERTS_std_dB, "[dB]")
    # Print Run Time
    print("Inference Time:", t)

    return [MSE_ERTS_linear_arr, MSE_ERTS_linear_avg, MSE_ERTS_dB_avg, ERTS_out]



