import torch
import torch.nn as nn

class BidirectionalEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional=True):
        super(self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, bidirectional=bidirectional)

    def forward(self, inputs):
        return self.rnn(inputs)

class BahdanauAttentionDecoder(nn.Module):
    def __init__(self, hidden_size):
        super(self).__init__()
        self.alignment_weight = nn.Linear(hidden_size, hidden_size)


    def forward(self, encoder_hidden, decoder_hidden):
         

        
        
