from pathlib import Path
import sys
script_dir = Path(__file__).resolve().parent.parent.parent
if str(script_dir) not in sys.path:
    sys.path.append(str(script_dir))
import torch
import torch.nn as nn
import torch.optim as optim
from Tools.utils import *

class VariableLengthGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(VariableLengthGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, lengths):
        # Sort input sequences by length
        lengths, sort_idx = torch.sort(torch.tensor(lengths), descending=True)
        x = x[sort_idx]
        
        # Pack padded sequence
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=True)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.batch_sizes[0], self.hidden_size).to(x.data.device)
        
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

# Example usage
input_size = 3  # size of each element in the sequence
hidden_size = 20  # size of hidden state
num_layers = 10  # number of layers in the GRU   
output_size = 3  # size of the output

[train_set,CV_set, _] =  torch.load(system_config.Dataset_path)
[_ ,_ , test_set] =  torch.load(system_config.test_set_path)

# Dummy data generation for demonstration
def get_batch(set):
    batch = []
    lengths = []
    target = []
    for traj in set:
        seq_length = traj.traj_length
        sequence = traj.y.squeeze().T #GRU expected Sequence_Length x Input_Size
        target.append(traj.x_real[-3:,0].squeeze())
        batch.append(sequence)
        lengths.append(seq_length)
    padded_batch = nn.utils.rnn.pad_sequence(batch, batch_first=True)
    return padded_batch, torch.tensor(lengths), torch.stack(target)

# Instantiate the model
model = VariableLengthGRU(input_size, hidden_size, num_layers, output_size)

# Dummy target for demonstration

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

if False:
    # Training loop
    num_epochs = 1000
    best_loss = float('inf')  # Initialize best loss to infinity
    for epoch in range(num_epochs):

        #Train
        model.train()
        optimizer.zero_grad()
        train_input, train_lengths ,train_target = get_batch(train_set)
        train_outputs = model(train_input, train_lengths)
        train_loss = criterion(train_outputs, train_target)
        train_loss.backward()
        optimizer.step()

        #CV
        model.eval()
        CV_input, CV_lengths ,CV_target = get_batch(CV_set)
        CV_outputs = model(CV_input, CV_lengths)
        CV_loss = criterion(CV_outputs, CV_target)


        # Print average loss for the epoch
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {10*torch.log(train_loss)} [dB] , CV Loss {10*torch.log(CV_loss)} ")

        if CV_loss < best_loss:
            best_loss = CV_loss
            torch.save(model.state_dict(), 'BiRNN_best_model.pth')
            print(f"Model saved with loss: {10*torch.log(best_loss)} [dB]")

    print("Training finished.")

# Load the model state dictionary
model.load_state_dict(torch.load('BiRNN_best_model.pth'))

# Set the model to evaluation mode
model.eval()
eval_set = test_set
input_batch, input_lengths ,target = get_batch(eval_set)
outputs = model(input_batch, input_lengths).detach()
BiRNN_estimated_energy_error = list()
MP_estimated_energy_error = list()

real_energy = list()

for i in range(outputs.shape[0]):
    estimated_energy = get_energy_from_velocities(outputs[i,0],outputs[i,1],outputs[i,2])
    real_energy.append(eval_set[i].init_energy)
    BiRNN_estimated_energy_error.append(100*(real_energy[-1]-estimated_energy)/real_energy[-1])
    _,est_para = get_mx_0(eval_set[i].y.squeeze(-1))
    MP_estimated_energy_error.append(100*(real_energy[-1]-est_para['init_energy'])/real_energy[-1])

plt.scatter(real_energy,MP_estimated_energy_error,s=2,label="MP Estimation")
plt.scatter(real_energy,BiRNN_estimated_energy_error,s=2,label="BiRNN Estimation")
plt.title("Energy Error Estimation")
plt.xlabel("Energy [MeV]")
plt.ylabel("Error [%]")
plt.legend()
plt.grid()
plt.show()



