from pathlib import Path
import sys
script_dir = Path(__file__).resolve().parent.parent.parent
if str(script_dir) not in sys.path:
    sys.path.append(str(script_dir))
import torch
import torch.nn as nn
import torch.optim as optim
from Tools.utils import *

class TransformerEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(TransformerEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        
        # Transformer encoder layers
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=3, dim_feedforward=hidden_size)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        
        # Output layer
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, src):
        src = src.permute(1, 0, 2)  # (batch_size, seq_len, input_size) -> (seq_len, batch_size, input_size)
        
        # Transformer encoder
        src = self.transformer_encoder(src)
        
        # Output layer
        output = self.fc(src[-1])  # Take the last hidden state
        
        return output

# Example usage
input_size = 3  # size of each element in the sequence
hidden_size = 32  # size of hidden state
num_layers = 10  # number of transformer encoder layers
output_size = 3  # size of the output

[train_set,CV_set, _] =  torch.load(system_config.Dataset_path)
[_ ,_ , test_set] =  torch.load(system_config.test_set_path)

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
model = TransformerEncoder(input_size, hidden_size, num_layers, output_size)

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
        train_outputs = model(train_input)
        train_loss = criterion(train_outputs, train_target)
        train_loss.backward()
        optimizer.step()

        #CV
        model.eval()
        CV_input, CV_lengths ,CV_target = get_batch(CV_set)
        CV_outputs = model(CV_input)
        CV_loss = criterion(CV_outputs, CV_target)


        # Print average loss for the epoch
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {10*torch.log(train_loss)} [dB] , CV Loss {10*torch.log(CV_loss)} ")

        if CV_loss < best_loss:
            best_loss = CV_loss
            torch.save(model.state_dict(), 'Transformer_best_model.pth')
            print(f"Model saved with loss: {10*torch.log(best_loss)} [dB]")

    print("Training finished.")

# Load the model state dictionary
model.load_state_dict(torch.load('Transformer_best_model.pth'))

# Set the model to evaluation mode
model.eval()
eval_set = test_set
input_batch, input_lengths ,target = get_batch(eval_set)
outputs = model(input_batch).detach()
TransformerEncoder_estimated_energy_error = list()
MP_estimated_energy_error = list()

real_energy = list()

for i in range(outputs.shape[0]):
    estimated_energy = get_energy_from_velocities(outputs[i,0],outputs[i,1],outputs[i,2])
    real_energy.append(eval_set[i].init_energy)
    TransformerEncoder_estimated_energy_error.append(100*torch.abs(real_energy[-1]-estimated_energy)/real_energy[-1])
    _,est_para = get_mx_0(eval_set[i].y.squeeze(-1))
    MP_estimated_energy_error.append(100*torch.abs(real_energy[-1]-est_para['init_energy'])/real_energy[-1])

plt.scatter(real_energy,MP_estimated_energy_error,s=2,label="MP Estimation")
plt.scatter(real_energy,TransformerEncoder_estimated_energy_error,s=2,label="TransformerEncoder Estimation")
plt.title("Energy Error Estimation")
plt.xlabel("Energy [MeV]")
plt.ylabel("Error [%]")
plt.legend()
plt.grid()
plt.show()
