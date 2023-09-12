import torch
import torch.nn as nn
from torch.autograd import Variable

class LSTM_model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, device):
        super(LSTM_model, self).__init__()
        self.input_size = input_size  # number of features
        self.hidden_size = hidden_size  # number of features in hidden state
        self.num_layers = num_layers  # number of stacked LSTM layers
        self.num_classes = num_classes  # number of output classes

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)  # LSTM layer
        self.fc = nn.Linear(hidden_size, num_classes)  # fully connected last layer
        self.device = device

    def forward(self, x):
        h0 = Variable(torch.zeros(self.num_layers, x.size(0),
                                  self.hidden_size)).to(self.device)  # hidden state
        c0 = Variable(torch.zeros(self.num_layers, x.size(0),
                                  self.hidden_size)).to(self.device)  # internal state
        # propagating input through LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (seq_length, batch_size, hidden_size)
        # decoding hidden state of last time step
        out = self.fc(out[:, -1, :])
        return out