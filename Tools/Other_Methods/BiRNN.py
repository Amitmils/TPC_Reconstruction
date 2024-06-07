from pathlib import Path
import sys
script_dir = Path(__file__).resolve().parent.parent.parent
if str(script_dir) not in sys.path:
    sys.path.append(str(script_dir))
import torch
import torch.nn as nn
import torch.optim as optim
from Tools.utils import *

class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True,bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, output_size)
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
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=True)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers *2, x.batch_sizes[0], self.hidden_size).to(x.data.device)
        
        # Forward pass through GRU
        out, _ = self.gru(x, h0)
        
        # Unpack packed sequence
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        
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
    def __init__(self,mode,output_path="Simulations/Particle_Tracking/temp models",lr=1e-3,logger=None) -> None:
        self.mode = mode
        self.logger = logger
        self.output_path = output_path
        if self.mode == "obs":
            input_size = 3  # size of each element in the sequence
            hidden_size = 20  # size of hidden state
            num_layers = 10  # number of layers in the GRU   
            output_size = 3  # size of the output
        elif self.mode == "fmap":
            input_size = 72  # size of each element in the sequence
            hidden_size = 128  # size of hidden state
            num_layers = 10  # number of layers in the GRU
            output_size = 3  # size of the output
        else:
            input_size = 6  # size of each element in the sequence
            hidden_size = 20  # size of hidden state
            num_layers = 10  # number of layers in the GRU
            output_size = 3  # size of the output
        self.model = BiRNN(input_size, hidden_size, num_layers, output_size)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

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
            elif self.mode == "fw":
                sequence = traj.x_estimated_FW.squeeze().T
            elif self.mode == "fmap":
                sequence = torch.cat((traj.bw_fmap.squeeze().T, traj.x_estimated_BW.squeeze().T), dim=1)
            target.append(traj.generated_traj[-3:,0].squeeze())
            batch.append(sequence)
            lengths.append(seq_length)
        padded_batch = nn.utils.rnn.pad_sequence(batch, batch_first=True)
        return padded_batch, torch.tensor(lengths), torch.stack(target)
    
    def pipeline_print(self,s):
        if self.logger is None:
            print(s)
        else:
            self.logger.info(s)
    def get_one_epoch_loss(self,set):
        input, lengths ,target = self.parse_data(set)
        output = self.model(input, lengths)
        return self.criterion(get_energy_from_velocities(output[:,0],output[:,1],output[:,2]), get_energy_from_velocities(target[:,0],target[:,1],target[:,2]))

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
            train_loss = self.criterion(get_energy_from_velocities(train_outputs[:,0],train_outputs[:,1],train_outputs[:,2]), get_energy_from_velocities(train_target[:,0],train_target[:,1],train_target[:,2]))

            train_loss.backward()
            self.optimizer.step()

            #CV
            self.model.eval()
            CV_input, CV_lengths ,CV_target = self.parse_data(CV_set)
            CV_outputs = self.model(CV_input, CV_lengths)
            # CV_loss = criterion(CV_outputs, CV_target)
            CV_loss = self.criterion(get_energy_from_velocities(CV_outputs[:,0],CV_outputs[:,1],CV_outputs[:,2]), get_energy_from_velocities(CV_target[:,0],CV_target[:,1],CV_target[:,2]))

            # Print average loss for the epoch
            self.pipeline_print(f"Epoch [{epoch+1}/{n_epochs}], Train Loss: {10*torch.log10(train_loss)} [dB] , CV Loss {10*torch.log10(CV_loss)} ")

            if CV_loss < best_loss:
                best_loss = CV_loss
                torch.save(self.model.state_dict(), os.path.join(self.output_path,f'best-BiRNN_model_{file_suffix}.pt'))
                self.pipeline_print(f"Model saved with loss: {10*torch.log10(best_loss)} [dB]")
    
    def eval(self,eval_set,load_model_path=None):
        if load_model_path is None:
            load_model_path = os.path.join(self.output_path, 'best-BiRNN_model_FINAL.pt')
        state_dict = torch.load(load_model_path)
        self.pipeline_print(f"Loading BiRNN Model {load_model_path}")
        self.model.load_state_dict(state_dict)
        self.model.eval()
        input_batch, input_lengths , target = self.parse_data(eval_set)
        outputs = self.model(input_batch, input_lengths).detach()
        #add results to set
        for i in range(len(eval_set)):
            eval_set[i].BiRNN_output = outputs[i]
        
        return self.criterion(get_energy_from_velocities(outputs[:,0],outputs[:,1],outputs[:,2]), get_energy_from_velocities(target[:,0],target[:,1],target[:,2]))

    def plot_data(self,eval_set,save_path):
        BiRNN_estimated_energy_error = list()
        MP_estimated_energy_error = list()
        real_energy = list()

        for i in range(len(eval_set)):
            estimated_energy = get_energy_from_velocities(eval_set[i].BiRNN_output[0],eval_set[i].BiRNN_output[1],eval_set[i].BiRNN_output[2])
            real_energy.append(eval_set[i].init_energy)
            BiRNN_estimated_energy_error.append(100*(estimated_energy-real_energy[-1])/real_energy[-1])
            _,est_para = get_mx_0(eval_set[i].y.squeeze(-1))
            MP_estimated_energy_error.append(100*(est_para['init_energy']-real_energy[-1])/real_energy[-1])

        plt.scatter(real_energy,MP_estimated_energy_error,s=2,label="MP Estimation")
        plt.scatter(real_energy,BiRNN_estimated_energy_error,s=2,label="BiRNN Estimation")
        plt.title(f"Energy Error Estimation {self.mode}\nAbs Avg {torch.tensor(BiRNN_estimated_energy_error).abs().mean()}, STD {torch.tensor(BiRNN_estimated_energy_error).std()}")
        plt.xlabel("Energy [MeV]")
        plt.ylabel("Error [%]")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(save_path,f"BiRNN_Head_Output_{self.mode}.png"))


if __name__ == "__main__":
    MODE = "fmap"
    [train_set,CV_set,test_set] =  torch.load("Simulations/Particle_Tracking/data/FC_PoC_Data_X.pt")
    Pipeline = BiRNNPipeLine(MODE)
    Pipeline.train(train_set=train_set,CV_set=CV_set,n_epochs=1500)
    Pipeline.eval(test_set)
    Pipeline.plot_data(test_set,"Tools/Other_Methods")




