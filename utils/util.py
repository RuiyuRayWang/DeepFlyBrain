import h5py
import torch
import numpy as np
from model.DeepFlyBrainModel import DeepFlyBrainModel


def generate_input_data():
    """
    Generates input data for the model.
    The size of the input object is (500, 4), representing a onehot encoded DNA sequence of length 500 bp.
    """
    # Generate random onehot encoded DNA sequence
    input_data = np.random.randint(0, 2, (500, 4))
    return input_data


def load_weights(model, h5_file_path):
    with h5py.File(h5_file_path, 'r') as f:
        # Load weights for each layer
        model.conv1d_1.weight.data = torch.tensor(np.array(f['conv1d_1']['conv1d_1']['kernel:0'])).permute(2, 1, 0)
        model.conv1d_1.bias.data = torch.tensor(np.array(f['conv1d_1']['conv1d_1']['bias:0']))
        
        model.time_distributed_1.weight.data = torch.tensor(np.array(f['time_distributed_1']['time_distributed_1']['kernel:0'])).permute(1, 0)
        model.time_distributed_1.bias.data = torch.tensor(np.array(f['time_distributed_1']['time_distributed_1']['bias:0']))
        
        model.lstm_1.weight_ih_l0.data = torch.tensor(np.array(f['bidirectional_1']['bidirectional_1']['forward_lstm_1']['kernel:0'])).permute(1, 0)
        model.lstm_1.weight_hh_l0.data = torch.tensor(np.array(f['bidirectional_1']['bidirectional_1']['forward_lstm_1']['recurrent_kernel:0'])).permute(1, 0)
        model.lstm_1.bias_ih_l0.data = torch.tensor(np.array(f['bidirectional_1']['bidirectional_1']['forward_lstm_1']['bias:0'])[:128])
        model.lstm_1.bias_hh_l0.data = torch.tensor(np.array(f['bidirectional_1']['bidirectional_1']['forward_lstm_1']['bias:0'])[128:])
        
        model.lstm_1.weight_ih_l0_reverse.data = torch.tensor(np.array(f['bidirectional_1']['bidirectional_1']['backward_lstm_1']['kernel:0'])).permute(1, 0)
        model.lstm_1.weight_hh_l0_reverse.data = torch.tensor(np.array(f['bidirectional_1']['bidirectional_1']['backward_lstm_1']['recurrent_kernel:0'])).permute(1, 0)
        model.lstm_1.bias_ih_l0_reverse.data = torch.tensor(np.array(f['bidirectional_1']['bidirectional_1']['backward_lstm_1']['bias:0'])[:128])
        model.lstm_1.bias_hh_l0_reverse.data = torch.tensor(np.array(f['bidirectional_1']['bidirectional_1']['backward_lstm_1']['bias:0'])[128:])
        
        model.dense_2.weight.data = torch.tensor(np.array(f['dense_2']['dense_2']['kernel:0'])).permute(1, 0)
        model.dense_2.bias.data = torch.tensor(np.array(f['dense_2']['dense_2']['bias:0']))
        
        model.dense_3.weight.data = torch.tensor(np.array(f['dense_3']['dense_3']['kernel:0'])).permute(1, 0)
        model.dense_3.bias.data = torch.tensor(np.array(f['dense_3']['dense_3']['bias:0']))

# Example usage:
# model = DeepFlyBrainModel()
# h5_file_path = "/mnt/WKD0Q2WQ/PROJECTS/D1D2_ENHANCER/data/public/DeepFlyBrain/DeepFlyBrain.hdf5"
# load_weights(model, h5_file_path)