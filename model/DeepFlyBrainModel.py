import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DeepFlyBrainModel(nn.Module):
    def __init__(self):
        super(DeepFlyBrainModel, self).__init__()
        self.conv1d_1 = nn.Conv1d(in_channels=4, out_channels=1024, kernel_size=24, stride=1, padding='same')
        self.max_pooling1d_1 = nn.MaxPool1d(kernel_size=12, stride=12)
        self.dropout_1 = nn.Dropout(0.5)
        self.time_distributed_1 = nn.Linear(1024, 128)
        self.lstm_1 = nn.LSTM(input_size=128, hidden_size=128, num_layers=1, batch_first=True, dropout=0.2, bidirectional=True)
        # self.dropout_2 = nn.Dropout(0.5)
        # self.flatten = nn.Flatten()
        # self.dense_2 = nn.Linear(128 * 2 * 41, 256)  # Adjust input size based on the output of LSTM
        # self.dropout_3 = nn.Dropout(0.5)
        # self.dense_3 = nn.Linear(256, 81)
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1d_1(x))
        x = self.max_pooling1d_1(x)
        x = self.dropout_1(x)
        x = x.permute(0, 2, 1)
        x = F.relu(self.time_distributed_1(x))
        x, _ = self.lstm_1(x)
        # x = self.dropout_2(x)
        # x = self.flatten(x)
        # x = F.relu(self.dense_2(x))
        # x = self.dropout_3(x)
        # x = torch.sigmoid(self.dense_3(x))
        return x

def generate_input_data():
    """
    Generates input data for the model.
    The size of the input object is (500, 4), representing a onehot encoded DNA sequence of length 500 bp.
    """
    # Generate random onehot encoded DNA sequence
    input_data = np.random.randint(0, 2, (500, 4))
    return input_data

# Example usage:
# model = DeepFlyBrainModel()
# X = torch.tensor(generate_input_data(), dtype=torch.float32).unsqueeze(0)  # Add batch dimension
# output = model(X)