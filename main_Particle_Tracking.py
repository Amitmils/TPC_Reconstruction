from pathlib import Path
import sys
script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.append(str(script_dir))
import torch
from Simulations.Extended_sysmdl import SystemModel
import os
from Pipelines.Pipeline_ERTS import Pipeline_ERTS as Pipeline
from Pipelines.Pipeline_concat_models import Pipeline_twoRTSNets
from datetime import datetime
from RTSNet.RTSNet_nn import RTSNetNN
from Tools.utils import *




system_config= CONFIG("Simulations/Particle_Tracking/config.yaml")
system_config.logger = setup_logger(os.path.join(system_config.path_results,"temp_log.log")) #this logger will be in config and in Pipeline itself
system_config.logger.info("Pipeline Start")

################
### Get Time ###
################
today = datetime.today()
now = datetime.now()
strToday = today.strftime("%m.%d.%y")
strNow = now.strftime("%H:%M:%S")
strTime = strToday + "_" + strNow
system_config.logger.info("Current Time = %s", strTime)

##########################
### Parameter settings ###
##########################
if system_config.use_cuda:
   if torch.cuda.is_available():
      device = torch.device('cuda')
      system_config.logger.info("Using GPU")
      torch.set_default_dtype(torch.float32)  # Set default data type
      torch.set_default_device('cuda')  # Set default device (optional)
   else:
      raise Exception("No GPU found, please set args.use_cuda = False")
else:
   device = torch.device('cpu')
   torch.set_default_device('cpu')  # Set default device (optional)
   system_config.logger.info("Using CPU")


####################
###  load data   ###
####################
system_config.logger.info(f"Train / CV Load : {os.path.basename(system_config.Dataset_path)}")
system_config.logger.info(f"Test Load : {os.path.basename(system_config.test_set_path)}")
[train_set,CV_set, _] =  torch.load(system_config.Dataset_path)
[_ ,_ , test_set] =  torch.load(system_config.test_set_path)

 

#Extract Relevant Data on the Trajectories
system_config.data_source = train_set[0].data_src
system_config.FTT_delta_t = train_set[0].delta_t

#We can set a smaller set in config
system_config.train_set_size = min(len(train_set),system_config.train_set_size)
system_config.CV_set_size = min(len(CV_set),system_config.CV_set_size)
system_config.test_set_size = min(len(test_set),system_config.test_set_size)
# system_config.output_folder = os.path.join(system_config.path_results,"##"+today.strftime("%m_%d")+"##"+now.strftime("%H_%M_%S"))
# os.makedirs(os.path.join(system_config.output_folder,'models'))
# os.makedirs(os.path.join(system_config.output_folder,'results'))



train_set = train_set[:system_config.train_set_size]
CV_set = CV_set[:system_config.CV_set_size]
test_set = test_set[:system_config.test_set_size]


system_config.logger.info("trainset size: %d",len(train_set))
system_config.logger.info("cvset size: %d",len(CV_set))
system_config.logger.info("testset size: %d",len(test_set))

#######################
###  System model   ###
#######################
sys_model = SystemModel(f, h, system_config.state_vector_size, system_config.observation_vector_size)# parameters for GT

#######################
### Evaluate RTSNet ###
#######################

######################
## RTSNet - 1 full ###
######################

## Build Neural Network
RTSNet_model = RTSNetNN()
RTSNet_model.NNBuild(sys_model,system_config)
# ## Train Neural Network
RTSNet_Pipeline = Pipeline(strTime, "RTSNet", "RTSNet",system_config)
RTSNet_Pipeline.setssModel(sys_model)
RTSNet_Pipeline.setModel(RTSNet_model)
RTSNet_Pipeline.logger = system_config.logger
RTSNet_Pipeline.logger.info("Number of trainable parameters for RTSNet: %d",sum(p.numel() for p in RTSNet_model.parameters() if p.requires_grad))
RTSNet_Pipeline.setTrainingParams()
if system_config.train == True:
   [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = RTSNet_Pipeline.NNTrain(sys_model, train_set, CV_set)
## Test Neural Network
RTSNet_Pipeline.NNTest(sys_model,test_set)
####################################################################################

if False:
################################
## RTSNet - 2 with full info ###
################################
   if load_dataset_for_pass2:
      print("Load dataset for pass 2")
      [train_input_pass2, train_target_pass2, cv_input_pass2, cv_target_pass2, test_input, test_target] = torch.load(DatasetPass1_path)
      
      print("Data Shape for RTSNet pass 2:")
      print("testset state x size:",test_target.size())
      print("testset observation y size:",test_input.size())
      print("trainset state x size:",train_target_pass2.size())
      print("trainset observation y size:",len(train_input_pass2),train_input_pass2[0].size())
      print("cvset state x size:",cv_target_pass2.size())
      print("cvset observation y size:",len(cv_input_pass2),cv_input_pass2[0].size())  
   else:
      ### save result of RTSNet1 as dataset for RTSNet2 
      RTSNet_model_pass1 = RTSNetNN()
      RTSNet_model_pass1.NNBuild(sys_model, args)
      RTSNet_Pipeline_pass1 = Pipeline(strTime, "RTSNet", "RTSNet",system_config)
      RTSNet_Pipeline_pass1.setssModel(sys_model)
      RTSNet_Pipeline_pass1.setModel(RTSNet_model_pass1)
      ### Optional to test it on test-set, just for checking
      print("Test RTSNet pass 1 on test set")
      [_, _, _,rtsnet_out_test,_] = RTSNet_Pipeline_pass1.NNTest(sys_model, test_input, test_target, path_results,load_model=True,load_model_path=RTSNetPass1_path)

      print("Test RTSNet pass 1 on training set")
      [_, _, _,rtsnet_out_train,_] = RTSNet_Pipeline_pass1.NNTest(sys_model, train_input, train_target, path_results,load_model=True,load_model_path=RTSNetPass1_path)
      print("Test RTSNet pass 1 on cv set")
      [_, _, _,rtsnet_out_cv,_] = RTSNet_Pipeline_pass1.NNTest(sys_model, cv_input, cv_target, path_results,load_model=True,load_model_path=RTSNetPass1_path)
      

      train_input_pass2 = rtsnet_out_train
      train_target_pass2 = train_target
      cv_input_pass2 = rtsnet_out_cv
      cv_target_pass2 = cv_target

      torch.save([train_input_pass2, train_target_pass2, cv_input_pass2, cv_target_pass2, test_input, test_target], DatasetPass1_path)
   #######################################
   ## RTSNet_2passes with full info   
   # Build Neural Network
   print("RTSNet(with full model info) pass 2 pipeline start!")
   RTSNet_model = RTSNetNN()
   RTSNet_model.NNBuild(sys_model_pass2, args)
   print("Number of trainable parameters for RTSNet pass 2:",sum(p.numel() for p in RTSNet_model.parameters() if p.requires_grad))
   ## Train Neural Network
   RTSNet_Pipeline = Pipeline(strTime, "RTSNet", "RTSNet_pass2")
   RTSNet_Pipeline.setssModel(sys_model_pass2)
   RTSNet_Pipeline.setModel(RTSNet_model)
   RTSNet_Pipeline.setTrainingParams(args)
   #######################################
   [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = RTSNet_Pipeline.NNTrain(sys_model_pass2, cv_input_pass2, cv_target_pass2, train_input_pass2, train_target_pass2, path_results)
   RTSNet_Pipeline.save()
   print("RTSNet pass 2 pipeline end!")
   #######################################
   # load trained Neural Network
   print("Concat two RTSNets and test")
   RTSNet_model1 = torch.load(RTSNetPass1_path)
   RTSNet_model2 = torch.load('RTSNet/best-model.pt')
   ## Set up Neural Network
   RTSNet_Pipeline_2passes = Pipeline_twoRTSNets(strTime, "RTSNet", "RTSNet")
   RTSNet_Pipeline_2passes.setModel(RTSNet_model1, RTSNet_model2)
   NumofParameter = RTSNet_Pipeline_2passes.count_parameters()
   print("Number of parameters for RTSNet with 2 passes: ",NumofParameter)
   ## Test Neural Network   
   [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg,rtsnet_out,RunTime] = RTSNet_Pipeline_2passes.NNTest(sys_model, test_input, test_target, path_results)







