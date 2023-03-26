import torch
import torch.nn as nn

class RNNLayer(nn.Module):
    def __init__(self, embedding_size, hidden_size):
        super(RNNLayer, self).__init__()
        self.hidden_size = hidden_size
        # Weight Matrix for current input
        self.W = torch.rand(1, embedding_size)
        
        # Weight Matrix for Previous Hidden State
        self.U = torch.rand(hidden_size, hidden_size)

        # initial Hidden State
        self.h_prev = torch.rand(1, self.hidden_size)

    def forward(self, inputs):
        # Output tensor of shape (input_length x hidden_size)
        h_out = torch.zeros(inputs.size(-2), self.hidden_size)

        for i, item in enumerate(inputs):
            # h_prev has shape = 1 x hidden_size
            # U has shape = hidden_size x hidden_size
            # Product to be calculated is U x h
            # Hence take transpose of h_prev
            uh = self.U @ torch.transpose(self.h_prev, 0, 1)
            # Take transpose of output to get 
            # shape of single output array = 1 x hidden_size
            uh = torch.transpose(uh, 0, 1)

            # item has shape = 1 x embedding_size
            # W has shape = hidden_size x embedding_size
            # Product to be calculated is W x item
            # Hence take transpose of item
            wx = self.W @ item.view(-1, 1)
            # Take transpose of output to get 
            # shape of single output array = 1 x hidden_size
            wx = torch.transpose(wx, 0, 1)

            # Add To get current hidden state
            h_current = uh + wx

            # Assign value to output
            h_out[i] = h_current

            # Designate current hidden state as previous
            self.h_prev = h_current

        return h_out  





