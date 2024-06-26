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
from Tools.Other_Methods.BiRNN import BiRNNPipeLine
system_config= CONFIG("Simulations/Particle_Tracking/config.yaml")
system_config.logger = setup_logger(os.path.join(system_config.path_results,"temp_log.log")) #this logger will be in config and in Pipeline itself
system_config.logger.info("Pipeline Start")
# warnings.simplefilter("error", UserWarning)

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
[train_set,CV_set, _] =  torch.load(system_config.Dataset_path,map_location=device)
[_ ,_ , test_set] =  torch.load(system_config.test_set_path,map_location=device)

#Extract Relevant Data on the Trajectories
system_config.data_source = train_set[0].data_src
system_config.FTT_delta_t = train_set[0].delta_t

#We can set a smaller set in config
system_config.train_set_size = min(len(train_set),system_config.train_set_size)
system_config.CV_set_size = min(len(CV_set),system_config.CV_set_size)
system_config.test_set_size = min(len(test_set),system_config.test_set_size)
os.makedirs(os.path.join(system_config.path_results,'temp models'),exist_ok=True)

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

## Build Neural Networks

RTSNet_Models = RTSNetNN()
RTSNet_Models.NNBuild(sys_model,system_config)
# ## Train Neural Network
RTSNet_Pipelines = Pipeline(strTime, "RTSNet", f"RTSNet",system_config)
RTSNet_Pipelines.setssModel(sys_model)
RTSNet_Pipelines.setModel(RTSNet_Models)
RTSNet_Pipelines.logger = system_config.logger
RTSNet_Pipelines.logger.info(f"Number of trainable parameters for RTSNet: %d",sum(p.numel() for p in RTSNet_Models.parameters() if p.requires_grad))
BiRNN_Pipeline = BiRNNPipeLine(mode="bw",output_path=os.path.join(system_config.path_results,"temp models"),lr = system_config.BiRNN_lr,logger=system_config.logger,device=device)
RTSNet_Pipelines.setHeadPipeline(BiRNN_Pipeline)
RTSNet_Pipelines.setTrainingParams()
if system_config.train == True:
   if system_config.train_RTS:
      [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = RTSNet_Pipelines.NNTrain(sys_model, train_set, CV_set,run_num = 0)
   if system_config.train_BiRNN:
      RTSNet_Pipelines.NNEval(sys_model,train_set ,load_RTS_model_path=system_config.RTS_model_path,load_BiRNN_model_path=system_config.BiRNN_model_path,set_name = "Train")
      RTSNet_Pipelines.NNEval(sys_model,CV_set ,load_RTS_model_path=system_config.RTS_model_path,load_BiRNN_model_path=system_config.BiRNN_model_path,set_name = "CV")
      RTSNet_Pipelines.head_pipeline.train(train_set=train_set,CV_set=CV_set,n_epochs=system_config.BiRNN_n_epochs)

saved_dir = RTSNet_Pipelines.NNEval(sys_model,test_set,load_RTS_model_path=system_config.RTS_model_path,load_BiRNN_model_path=system_config.BiRNN_model_path,set_name = "Test")


torch.save([train_set,CV_set,test_set],os.path.join(saved_dir,'traj_outputs.pt'))
print(f"Run output saved to {saved_dir}")
####################################################################################




