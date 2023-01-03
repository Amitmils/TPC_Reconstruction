import torch
from datetime import datetime

from Smoothers.EKF_test import EKFTest
from Smoothers.Extended_RTS_Smoother_test import S_Test
from Smoothers.ParticleSmoother_test import PSTest

from Simulations.Extended_sysmdl import SystemModel
from Simulations.utils import DataGen,Short_Traj_Split
import Simulations.config as config
from Simulations.Lorenz_Atractor.parameters import m1x_0, m2x_0, m, n,\
f, h, h_nonlinear, Q_structure, R_structure

from Pipelines.Pipeline_ERTS import Pipeline_ERTS as Pipeline
from Pipelines.Pipeline_concat_models import Pipeline_twoRTSNets
from Pipelines.Pipeline_ERTS_multipass import Pipeline_ERTS as Pipeline_multipass

from RTSNet.RTSNet_nn import RTSNetNN
from RTSNet.RTSNet_nn_multipass import RTSNetNN_multipass

from Plot import Plot_extended as Plot

print("Pipeline Start")
################
### Get Time ###
################
today = datetime.today()
now = datetime.now()
strToday = today.strftime("%m.%d.%y")
strNow = now.strftime("%H:%M:%S")
strTime = strToday + "_" + strNow
print("Current Time =", strTime)

###################
###  Settings   ###
###################
args = config.general_settings()
args.N_E = 1000
args.N_CV = 100
args.N_T = 200
args.T = 20
args.T_test = 20
### settings for RTSNet - 1 (later will change for RTSNet - 2)
args.in_mult_KNet = 40
args.out_mult_KNet = 5
args.in_mult_RTSNet = 40
args.out_mult_RTSNet = 5

offset = 0
chop = False
sequential_training = False
path_results = 'RTSNet/'
DatafolderName = 'Simulations/Lorenz_Atractor/data/T20_hNL' + '/'

r2 = torch.tensor([1e-3]) # [10, 1, 0.1, 0.01, 1e-3]
vdB = 0 # ratio v=q2/r2
v = 10**(vdB/10)
q2 = torch.mul(v,r2)

Q = q2[0] * Q_structure
R = r2[0] * R_structure

print("1/r2 [dB]: ", 10 * torch.log10(1/r2[0]))
print("1/q2 [dB]: ", 10 * torch.log10(1/q2[0]))

traj_resultName = ['traj_lorDT_NLobs_rq3030_T20.pt']
dataFileName = ['data_lor_v0_rq3030_T20.pt']

#########################################
###  Generate and load data DT case   ###
#########################################

sys_model = SystemModel(f, Q, h_nonlinear, R, args.T, args.T_test, m, n)# parameters for GT
sys_model.InitSequence(m1x_0, m2x_0)# x0 and P0
## Model with H=I          
sys_model_H = SystemModel(f, Q, h, R, args.T,args.T_test, m, n)
sys_model_H.InitSequence(m1x_0, m2x_0)

print("Start Data Gen")
DataGen(args, sys_model, DatafolderName + dataFileName[0],randomInit=False)
print("Data Load")
print(dataFileName[0])
[train_input_long,train_target_long, cv_input, cv_target, test_input, test_target] =  torch.load(DatafolderName + dataFileName[0])  
if chop: 
   print("chop training data")    
   [train_target, train_input, train_init] = Short_Traj_Split(train_target_long, train_input_long, args.T)
   # [cv_target, cv_input] = Short_Traj_Split(cv_target, cv_input, T)
else:
   print("no chopping") 
   train_target = train_target_long[:,:,0:args.T]
   train_input = train_input_long[:,:,0:args.T] 
   # cv_target = cv_target[:,:,0:T]
   # cv_input = cv_input[:,:,0:T]  

print("trainset size:",train_target.size())
print("cvset size:",cv_target.size())
print("testset size:",test_target.size())

######################################
### Evaluate Filters and Smoothers ###
######################################
### Evaluate EKF full
print("Evaluate EKF full")
[MSE_EKF_linear_arr, MSE_EKF_linear_avg, MSE_EKF_dB_avg, EKF_KG_array, EKF_out] = EKFTest(sys_model, test_input, test_target)

### Evaluate RTS full
print("Evaluate RTS full")
[MSE_ERTS_linear_arr, MSE_ERTS_linear_avg, MSE_ERTS_dB_avg, ERTS_out] = S_Test(args, sys_model, test_input, test_target)

### Evaluate Particle Smoother
print("Evaluate Particle Smoother full")
[MSE_PS_linear_arr, MSE_PS_linear_avg, MSE_PS_dB_avg, PS_out, t_PS] = PSTest(sys_model, test_input, test_target,N_FWParticles=100, M_BWTrajs=10, init_cond=None)


### Save trajectories
trajfolderName = 'Smoothers' + '/'
DataResultName = traj_resultName[0]
EKF_sample = torch.reshape(EKF_out[0,:,:],[1,m,args.T_test])
ERTS_sample = torch.reshape(ERTS_out[0,:,:],[1,m,args.T_test])
PS_sample = torch.reshape(PS_out[0,:,:],[1,m,args.T_test])
target_sample = torch.reshape(test_target[0,:,:],[1,m,args.T_test])
input_sample = torch.reshape(test_input[0,:,:],[1,n,args.T_test])
torch.save({
            'EKF': EKF_sample,
            'ERTS': ERTS_sample,
            'PS': PS_sample,
            'ground_truth': target_sample,
            'observation': input_sample,
            }, trajfolderName+DataResultName)

#######################
### Evaluate RTSNet ###
#######################
## RTSNet with full info
## Build Neural Network
print("RTSNet with full model info")
RTSNet_model = RTSNetNN()
CompositionLoss = True
RTSNet_model.NNBuild(sys_model, args)
## Train Neural Network
RTSNet_Pipeline = Pipeline(strTime, "RTSNet", "RTSNet")
RTSNet_Pipeline.setssModel(sys_model)
RTSNet_Pipeline.setModel(RTSNet_model)
print("Number of trainable parameters for RTSNet:",sum(p.numel() for p in RTSNet_model.parameters() if p.requires_grad))
RTSNet_Pipeline.setTrainingParams(n_steps=2000, n_Batch=100, learningRate=1e-4, weightDecay=1e-4) 
if(chop):
   [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = RTSNet_Pipeline.NNTrain(sys_model, cv_input, cv_target, train_input, train_target, path_results,randomInit=True,train_init=train_init)
else:
   print("Composition Loss:",CompositionLoss)
   [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = RTSNet_Pipeline.NNTrain(sys_model, cv_input, cv_target, train_input, train_target, path_results,CompositionLoss=CompositionLoss)
## Test Neural Network
[MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg,rtsnet_out,RunTime] = RTSNet_Pipeline.NNTest(sys_model, test_input, test_target, path_results)


###############################################
### Concat two RTSNets with model mismatch  ###
###############################################
args.in_mult_KNet = 5
args.out_mult_KNet = 40
args.in_mult_RTSNet = 5
args.out_mult_RTSNet = 40

### Train pass2 on the output of pass1
print("test pass1 on Train Set")
fileName = "Simulations/Lorenz_Atractor/data/T20_hNL/Pass1_rq3030_T20.pt"
# [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg,rtsnet_out_train,RunTime] = RTSNet_Pipeline.NNTest(sys_model, train_input, train_target, path_results)
# cv_input_pass2 = rtsnet_out_train[0:N_CV]
# cv_target_pass2 = train_target[0:N_CV]
# train_input_pass2 = rtsnet_out_train[N_CV:-1]
# train_target_pass2 = train_target[N_CV:-1]
# torch.save([train_input_pass2, train_target_pass2, cv_input_pass2, cv_target_pass2], fileName)
[train_input_pass2, train_target_pass2, cv_input_pass2, cv_target_pass2] = torch.load(fileName)

print("Train RTSNet pass2")
RTSNet_model = RTSNetNN()
RTSNet_model.NNBuild(sys_model_H, args)
## Train Neural Network
RTSNet_Pipeline = Pipeline(strTime, "RTSNet", "RTSNet")
RTSNet_Pipeline.setssModel(sys_model_H)
RTSNet_Pipeline.setModel(RTSNet_model)
print("Number of trainable parameters for RTSNet:",sum(p.numel() for p in RTSNet_model.parameters() if p.requires_grad))
RTSNet_Pipeline.setTrainingParams(n_steps=2000, n_Batch=100, learningRate=1e-4, weightDecay=1e-4) 
if(chop):
   [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = RTSNet_Pipeline.NNTrain(sys_model_H, cv_input_pass2, cv_target_pass2, train_input_pass2, train_target_pass2, path_results,randomInit=True,train_init=train_init)
else:
   [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = RTSNet_Pipeline.NNTrain(sys_model_H, cv_input_pass2, cv_target_pass2, train_input_pass2, train_target_pass2, path_results)


## load trained Neural Network
print("RTSNet with model mismatch")
RTSNet_model1 = torch.load('RTSNet/checkpoints/LorenzAttracotor/DT/HNL/rq3030_T20.pt')
RTSNet_model2 = torch.load('RTSNet/best-model.pt')
## Setup Pipeline
RTSNet_Pipeline = Pipeline_twoRTSNets(strTime, "RTSNet", "RTSNet")
RTSNet_Pipeline.setModel(RTSNet_model1, RTSNet_model2)
NumofParameter = RTSNet_Pipeline.count_parameters()
print("Number of parameters for RTSNet: ",NumofParameter)
## Test Neural Network
[MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg,rtsnet_out_2pass,RunTime] = RTSNet_Pipeline.NNTest(sys_model, test_input, test_target, path_results)


########################
### RTSNet Multipass ###
########################
## Build Neural Network
# print("RTSNet multipass")
# iterations = 2 # number of passes
# RTSNet_model = RTSNetNN_multipass(iterations)
# RTSNet_model.NNBuild_multipass(sys_model)
# ## Train Neural Network
# RTSNet_Pipeline = Pipeline_multipass(strTime, "RTSNet", "RTSNet")
# RTSNet_Pipeline.setModel(RTSNet_model)
# RTSNet_Pipeline.setTrainingParams(n_Epochs=2000, n_Batch=100, learningRate=1e-4, weightDecay=1e-4)
# NumofParameter = RTSNet_Pipeline.count_parameters()
# if(chop):
#    [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = RTSNet_Pipeline.NNTrain(sys_model, cv_input, cv_target, train_input, train_target, path_results,CompositionLoss=True,randomInit=True,train_init=train_init)
# else:
#    [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = RTSNet_Pipeline.NNTrain(sys_model, cv_input, cv_target, train_input, train_target, path_results,CompositionLoss=True)
# ## Test Neural Network
# [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg,rtsnet_out,RunTime] = RTSNet_Pipeline.NNTest(sys_model, test_input, test_target, path_results)
   





