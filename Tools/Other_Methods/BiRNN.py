from pathlib import Path
import sys
script_dir = Path(__file__).resolve().parent.parent.parent
if str(script_dir) not in sys.path:
    sys.path.append(str(script_dir))
import torch
import torch.nn as nn
import torch.optim as optim
from Tools.utils import *
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')

if torch.cuda.is_available():
  device = torch.device('cuda')
  print("Using GPU:", torch.cuda.get_device_name(torch.cuda.current_device()))
  torch.set_default_dtype(torch.float32)  # Set default data type
  torch.set_default_device('cuda')  # Set default device (optional)
  #Setting default device to 'cuda' causes some problems with the spline functions that try to turn tensors into numpy inside the functions
  #therefore, as a WA i set this to cpu before those functions. These functions dont need to be backproped through
else:
    device = torch.device('cpu')
    print("Using CPU")

class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size,is_bidirectional = True,many_to_many=False):
        super(BiRNN, self).__init__()
        self.is_bidirection = is_bidirectional
        self.many_to_many = many_to_many
        self.hidden_layer_multiplier = 2 if self.is_bidirection else 1
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True,bidirectional=self.is_bidirection)
        self.fc = nn.Linear(hidden_size*self.hidden_layer_multiplier, output_size)

        self.BiRNN_param = []
        for _,param in self.named_parameters():
            self.BiRNN_param.append(param)

    def set_requires_grad(self,requires):
        for param in self.parameters():
            param.requires_grad = requires

    def forward(self, x, lengths):
        # Sort input sequences by length
        lengths, sort_idx = torch.sort(lengths, descending=True)
        x = x[sort_idx]


        # Pack padded sequence
        x = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)

        # Initialize hidden state
        h0 = torch.zeros(self.num_layers * self.hidden_layer_multiplier, x.batch_sizes[0], self.hidden_size).to(x.data.device)

        # Forward pass through GRU
        out, _ = self.gru(x, h0)

        # Unpack packed sequence
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        if not(self.many_to_many):
            # Gather the last output for each sequence
            idx = (lengths - 1).view(-1, 1).expand(out.size(0), out.size(2)).unsqueeze(1)
            out = out.gather(1, idx).squeeze(1)

        # Pass through linear layer
        out = self.fc(out)
        # Reorder the output to match the original order of the input sequences
        _, unsort_idx = torch.sort(sort_idx)
        out = out[unsort_idx]
        return out

class BiRNNPipeLine():
    def __init__(self,mode,output_path="Models",lr=1e-3,logger=None,max_length = 1000,device=device) -> None:
        self.mode = mode
        self.logger = logger
        self.output_path = output_path
        self.device = device
        self.max_length = max_length
        if self.mode == "obs" or self.mode == "bw_pos" or self.mode == "bw_vel" or self.mode == "real_vel":
            input_size = 3  # size of each element in the sequence
        elif self.mode == "fmap" or self.mode == "bw_gain":
            input_size = 42  # size of each element in the sequence
        elif self.mode == "bw_dx" or self.mode == "bw_inov":
            input_size = 12  # size of each element in the sequence
        elif self.mode == "bw_fw_gain":
            input_size = 21  # size of each element in the sequence
        elif self.mode == "bw_pos_gain" or self.mode == 'bw_vel_gain':
            input_size = 39  # size of each element in the sequence
        elif self.mode == 'bw_vel_gain_2':
            input_size = 21  # size of each element in the sequence
        elif self.mode == 'real_energy' or self.mode == 'bw_energy':
            input_size = 1
        else:
            input_size = 6  # size of each element in the sequence
        self.output_size = 3 # size of the output
        hidden_size = 20  # size of hidden state
        num_layers = 10  # number of layers in the GRU
        self.model = BiRNN(input_size, hidden_size, num_layers, self.output_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def criterion(self,train_output,target):
        if self.output_size == 3:
          train_output = get_energy_from_velocities(train_output[:,0],train_output[:,1],train_output[:,2])
        target = get_energy_from_velocities(target[:,0],target[:,1],target[:,2])

        mse_loss = nn.MSELoss()
        # return ((((train_output-target).abs())/target)).mean()
        return mse_loss(train_output,target)
        # return ((((train_output-target)**2)/target)).mean()

    def parse_data(self,set):
        batch = []
        lengths = []
        target = []
        for traj in set:
          seq_length = traj.traj_length
          if self.mode == "obs":
              sequence = traj.y.squeeze().T #GRU expected Sequence_Length x Input_Size
          elif self.mode == "bw":
              sequence = traj.x_estimated_BW.squeeze().T
          elif self.mode == "gen":
              sequence = traj.generated_traj.squeeze().T
          elif self.mode == "real":
              sequence = traj.x_real.squeeze().T
          elif self.mode == "real_energy":
              velocities = traj.x_real.squeeze()[-3:,:].T
              sequence = get_energy_from_velocities(velocities[:,0],velocities[:,1],velocities[:,2]).reshape(-1,1)
          elif self.mode == "bw_energy":
              velocities = traj.x_estimated_BW.squeeze()[-3:,:].T
              sequence = get_energy_from_velocities(velocities[:,0],velocities[:,1],velocities[:,2]).reshape(-1,1)
          elif self.mode == "real_vel":
              sequence = traj.x_real.squeeze()[-3:,:].T
          elif self.mode == "fw":
              sequence = traj.x_estimated_FW.squeeze().T
          elif self.mode == "bw_dx":
              sequence = torch.cat((traj.bw_dx.squeeze().T, traj.x_estimated_BW.squeeze().T), dim=1)
          elif self.mode == "bw_inov":
              sequence = torch.cat((traj.bw_inov.squeeze().T, traj.x_estimated_BW.squeeze().T), dim=1)
          elif self.mode == "fmap":
              sequence = torch.cat((traj.fmap.squeeze().T[:,:36], traj.x_estimated_BW.squeeze().T), dim=1)
          elif self.mode == "bw_gain":
              sequence = torch.cat((traj.bw_gain.squeeze().T, traj.x_estimated_BW.squeeze().T), dim=1)
          elif self.mode == "bw_fw_gain":
              sequence = torch.cat((traj.fw_gain.squeeze().T, traj.x_estimated_BW.squeeze()[:3,:].T), dim=1)
          elif self.mode == "bw_pos":
              sequence = traj.x_estimated_BW.squeeze()[:3,:].T
          elif self.mode == "bw_pos_gain":
              sequence = torch.cat((traj.bw_gain.squeeze().T, traj.x_estimated_BW.squeeze()[:3,:].T), dim=1)
          elif self.mode == "bw_vel":
              sequence = traj.x_estimated_BW.squeeze()[-3:,:].T
          elif self.mode == "bw_vel_gain":
              sequence = torch.cat((traj.bw_gain.squeeze().T, traj.x_estimated_BW.squeeze()[-3:,:].T), dim=1)
          elif self.mode == "bw_vel_gain_2":
              #get only gains for velocity
              sequence = torch.cat((traj.bw_gain.squeeze()[18:].T, traj.x_estimated_BW.squeeze()[-3:,:].T), dim=1)
          elif self.mode == "birnn_smoother":
              sequence = traj.BiRNN_Smoother_output.squeeze().T
          target.append(traj.generated_traj[-3:,0].squeeze())
          batch.append(sequence[:min(seq_length,self.max_length)])
          lengths.append(min(seq_length,self.max_length))
        padded_batch = nn.utils.rnn.pad_sequence(batch, batch_first=True)
        return padded_batch.to(device), torch.tensor(lengths), torch.stack(target).to(device)

    def pipeline_print(self,s):
        if self.logger is None:
            print(s)
        else:
            self.logger.info(s)
    def get_one_epoch_loss(self,set):
        input, lengths ,target = self.parse_data(set)
        output = self.model(input, lengths)
        return self.criterion(output, target)

    def train(self,train_set=None,CV_set=None,n_epochs=3000,file_suffix="FINAL",data_path = None):
        if data_path is not None:
            [train_set,CV_set,_] =  torch.load(data_path)
        self.model.set_requires_grad(True)
        best_loss = float('inf')  # Initialize best loss to infinity
        for epoch in range(n_epochs):
            #Train
            self.model.train()
            self.optimizer.zero_grad()
            train_input, train_lengths ,train_target = self.parse_data(train_set)
            train_outputs = self.model(train_input, train_lengths)
            # train_loss = criterion(train_outputs, train_target)
            train_loss = self.criterion(train_outputs, train_target)

            train_loss.backward()
            self.optimizer.step()
            CV_input, CV_lengths ,CV_target = self.parse_data(CV_set)
            #CV
            self.model.eval()
            CV_outputs = self.model(CV_input, CV_lengths)
            # CV_loss = criterion(CV_outputs, CV_target)
            CV_loss = self.criterion(CV_outputs, CV_target)

            # Print average loss for the epoch
            self.pipeline_print(f"Epoch [{epoch+1}/{n_epochs}], Train Loss: {10*torch.log10(train_loss)} [dB] , CV Loss {10*torch.log10(CV_loss)} ")

            if CV_loss < best_loss:
                best_loss = CV_loss
                torch.save(self.model.state_dict(), os.path.join(self.output_path,f'best-BiRNN_model_{file_suffix}.pt'))
                self.pipeline_print(f"Model saved with loss: {10*torch.log10(best_loss)} [dB]")
        self.pipeline_print(f"Best Loss :  {10*torch.log10(best_loss)} [dB]")

    def eval(self,eval_set,load_model_path=None,file_suffix="FINAL"):
        if load_model_path is None:
            load_model_path = os.path.join(self.output_path, f'best-BiRNN_model_{file_suffix}.pt')
        state_dict = torch.load(load_model_path,map_location=device)
        self.pipeline_print(f"Loading BiRNN Model {load_model_path}")
        self.model.load_state_dict(state_dict)
        self.model.eval()
        input_batch, input_lengths , target = self.parse_data(eval_set)
        outputs = self.model(input_batch, input_lengths).detach()
        loss = self.criterion(outputs, target)
        self.pipeline_print(f"Loss : {10*torch.log10(loss)}")
        #add results to set
        for i in range(len(eval_set)):
            eval_set[i].BiRNN_output = outputs[i]
        return 10*torch.log10(loss)

    def plot_data(self,eval_set,save_path,suffix=None):
        BiRNN_estimated_energy_error = list()
        MP_estimated_energy_error = list()
        real_energy = list()

        for i in range(len(eval_set)):
            estimated_energy = get_energy_from_velocities(eval_set[i].BiRNN_output[0],eval_set[i].BiRNN_output[1],eval_set[i].BiRNN_output[2]) if self.output_size == 3 else eval_set[i].BiRNN_output.cpu()
            real_energy.append(eval_set[i].init_energy)
            BiRNN_estimated_energy_error.append((100*(estimated_energy-real_energy[-1])/real_energy[-1]).abs().cpu())
        plt.figure()
        plt.scatter(real_energy,BiRNN_estimated_energy_error,s=2,label="BiRNN Estimation")
        plt.title(f"Energy Error Estimation {self.mode}\nAbs Avg {torch.tensor(BiRNN_estimated_energy_error).mean()}, STD {torch.tensor(BiRNN_estimated_energy_error).std()}")
        plt.xlabel("Energy [MeV]")
        plt.ylabel("Error [%]")
        plt.legend()
        plt.grid()
        save_path = os.path.join(save_path,f"BiRNN_Head_Output_{self.mode if suffix is None else suffix}.png")
        if os.path.exists(save_path):
          os.remove(save_path)
        plt.savefig(save_path)
        return torch.tensor(BiRNN_estimated_energy_error).abs().mean(),torch.tensor(BiRNN_estimated_energy_error).std()

    def plot_data_multiple_BiRNN(self, eval_sets_list, save_path,suffix=None,labels_list=['BiRNN Error'],max_energy = 3 ,plot_steps = 0.2,y_axis = "%"):

        BiRNN_abs_avg_errors_list = list()
        BiRNN_std_errors_list = list()
        BiRNN_l2_errors_list = list()
        for eval_set in eval_sets_list:
          BiRNN_estimated_energy_error = list()
          real_energy = list()
          estimated_energy = list()
          for i in range(len(eval_set)):
              estimated_energy.append(get_energy_from_velocities(eval_set[i].BiRNN_output[0], eval_set[i].BiRNN_output[1], eval_set[i].BiRNN_output[2]).cpu() if self.output_size == 3 else eval_set[i].BiRNN_output.cpu())
              real_energy.append(eval_set[i].init_energy)
              BiRNN_estimated_energy_error.append((100 * (estimated_energy[-1] - real_energy[-1]) / real_energy[-1]).abs().cpu())


          real_energy = np.array(real_energy)
          estimated_energy = np.array(estimated_energy)
          BiRNN_estimated_energy_error = np.array(BiRNN_estimated_energy_error)

          # Define energy bins
          bins = np.arange(0.4, max_energy, plot_steps)  # Adjust the range and step as needed
          bin_centers = (bins[:-1] + bins[1:]) / 2

          BiRNN_abs_avg_errors = []
          BiRNN_std_errors = []
          BiRNN_l2_errors = []
          mse_loss = nn.MSELoss()
          for i in range(len(bins) - 1):
              bin_mask = (real_energy >= bins[i]) & (real_energy < bins[i + 1])

              if np.sum(bin_mask) > 0:
                  BiRNN_errors_in_bin = BiRNN_estimated_energy_error[bin_mask]
                  BiRNN_l2_errors.append(10*torch.log10(mse_loss(torch.tensor(estimated_energy[bin_mask]), torch.tensor(real_energy[bin_mask]))).item())
                  BiRNN_abs_avg_errors.append(np.mean(BiRNN_errors_in_bin))
                  BiRNN_std_errors.append(np.std(BiRNN_errors_in_bin))
              else:
                  BiRNN_abs_avg_errors.append(np.nan)
                  BiRNN_std_errors.append(np.nan)
                  BiRNN_l2_errors.append(np.nan)

          # Convert to numpy arrays for easier handling in plotting
          BiRNN_abs_avg_errors_list.append(np.array(BiRNN_abs_avg_errors))
          BiRNN_std_errors_list.append(np.array(BiRNN_std_errors))
          BiRNN_l2_errors_list.append(np.array(BiRNN_l2_errors))

        # Plotting
        plt.figure(figsize=(10, 6))

        # Plot
        for abs_avg_errors, std_errors,l2_errors,label in zip(BiRNN_abs_avg_errors_list, BiRNN_std_errors_list,BiRNN_l2_errors_list,labels_list):
          if y_axis == '%':
            plt.plot(bin_centers, abs_avg_errors, '-o', label=label)
          else:
            plt.plot(bin_centers, l2_errors, '-o', label=label)
          # plt.fill_between(bin_centers, abs_avg_errors - std_errors, BiRNN_abs_avg_errors + std_errors, alpha=0.2)
        plt.yscale('log')
        plt.xlabel('Real Energy')
        plt.ylabel('Absolute Average Error (%)' if y_axis == '%' else "MSE [dB]")
        plt.title(f"{'%' if y_axis == '%' else 'MSE'} Error Comparison\nStep Size {plot_steps}")
        plt.legend()
        plt.grid(True)
        save_path = os.path.join(save_path,f"BiRNN_Head_Output_{self.mode if suffix is None else suffix}.png")
        if os.path.exists(save_path):
          os.remove(save_path)
        plt.savefig(save_path)
        return torch.tensor(BiRNN_estimated_energy_error).mean(),torch.tensor(BiRNN_estimated_energy_error).std()



if __name__ == "__main__":
    modes = ['bw_vel','bw_energy']#['real_energy','real_vel','bw_vel','bw_vel_gain_2','bw_vel_gain','bw_vel_gain_2','bw','bw_gain']#,'bw_pos','bw_vel','bw_pos_gain','bw_vel_gain']#['bw_gain','bw_dx','bw_inov','bw','obs','real','fw']
    num_runs=10
    epochs = 5000
    run_name = "L2_0.5-1MeV_uniform_energy_error"
    data_path = "Tools/Other_Methods/FC_PoC_Data_Z_EnerKF.pt"
    root_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)),"BiRNN_HEAD_Output")
    models_save_path = os.path.join(root_folder,"Models",run_name)
    graphs_save_path = os.path.join(root_folder,"Graphs",run_name)
    results_save_path = os.path.join(root_folder,"Results_Summary")

    os.makedirs(models_save_path,exist_ok=True)
    os.makedirs(graphs_save_path,exist_ok=True)
    os.makedirs(results_save_path,exist_ok=True)

    [train_set,CV_set,test_set] =  torch.load(data_path)
    results = {"Mode" : [] , "Loss" : [] , "Abs Avg" : [] , "STD" : []}
    for run in range(num_runs):
        for mode in modes:
                print(f"\n######### {mode}_{run} #########")
                Pipeline = BiRNNPipeLine(mode,output_path=models_save_path,device=device,max_length=100)
                Pipeline.train(train_set=train_set,CV_set=CV_set,n_epochs=epochs,file_suffix=f"{mode}_{run}")
                loss = Pipeline.eval(test_set,file_suffix=f"{mode}_{run}")
                abs_avg , std = Pipeline.plot_data(test_set,save_path=graphs_save_path,suffix=f"{mode}_{run}") #plot_data_multiple_BiRNN
                print(f"abs avg : {abs_avg} , std {std}")
                results["Mode"].append(f"{mode}_{run}")
                results["Loss"].append(loss.cpu().item())
                results["Abs Avg"].append(abs_avg.cpu().item())
                results["STD"].append(std.cpu().item())

                df = pd.DataFrame(results)
                if os.path.exists(f"Results_{run_name}.csv"):
                            os.remove(f"Results_{run_name}.csv")
                df.to_csv(os.path.join(results_save_path,f"Results_{run_name}.csv"))
