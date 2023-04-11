import torch
import torch.nn as nn


class RNNLayer(nn.Module):
    def __init__(self, embedding_size, hidden_size, activation='relu'):
        super(RNNLayer, self).__init__()
        self.hidden_size = hidden_size

        # Weight Matrix for current input
        self.W = torch.randn(hidden_size, embedding_size)

        # Weight Matrix for Previous Hidden State
        self.U = torch.randn(hidden_size, hidden_size)

        # Initial Hidden State
        self.h_prev = torch.randn(1, self.hidden_size)

        # Activation Function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()

    def forward(self, inputs, h_0=None):
        # Assign custom hidden state if any
        if h_0 is not None:
            self.h_prev = h_0

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
            h_out[i] = self.activation(h_current)

            # Designate current hidden state as previous
            self.h_prev = h_current

        return h_out


class RNNLayerForTextClassification(RNNLayer):
    def __init__(self, embedding_size, hidden_size, output_dim):
        super(RNNLayer, self).__init__()
        self.rnnlayer = RNNLayer(embedding_size, hidden_size)
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        if self.output_dim == 1:
            self.activation = nn.Sigmoid()
        elif self.output_dim >= 3:
            self.activation = nn.Softmax(dim=1)

        # Final Layer Weights for Classification
        self.V = torch.rand(output_dim, self.hidden_size)

    def forward(self, inputs):
        h_out = self.rnnlayer(inputs)
        h_final = h_out[-1]
        y = self.V @ h_final.view(-1, 1)
        return self.activation(y)


class LSTM(nn.Module):
    def __init__(self, embedding_size, hidden_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size

        # Initial Hidden State
        self.h_prev = torch.randn(1, self.hidden_size)

        # Initial Context Vector
        self.c_prev = torch.randn(1, self.hidden_size)

        # Weight Matrices for the Forget Gate
        self.U_f = torch.randn(hidden_size, hidden_size)
        self.W_f = torch.randn(hidden_size, embedding_size)

        # Sigmoid function
        self.sigmoid = nn.Sigmoid()

        # Weight Matrices for the actual information
        self.U_g = torch.randn(hidden_size, hidden_size)
        self.W_g = torch.randn(hidden_size, embedding_size)

        # Tanh function
        self.tanh = nn.Tanh()

        # Weight Matrices for Add Gate
        self.U_i = torch.randn(hidden_size, hidden_size)
        self.W_i = torch.randn(hidden_size, embedding_size)

        # Weight Matrices for Output Gate
        self.U_o = torch.randn(hidden_size, hidden_size)
        self.W_o = torch.randn(hidden_size, embedding_size)

    def forward(self, inputs, h_0=None, c_0=None):
        # Assign custom hidden state if any
        if h_0 is not None:
            self.h_prev = h_0

        # Assign custom context vector if any
        if c_0 is not None:
            self.c_prev = c_0

        # Output tensor of hidden states
        # with shape (input_length x hidden_size)
        h_out = torch.zeros(inputs.size(-2), self.hidden_size)

        # Output tensor of context vectors
        # with shape (input_length x hidden_size)
        c_out = torch.zeros(inputs.size(-2), self.hidden_size)

        for i, item in enumerate(inputs):

            # h_prev has shape = 1 x hidden_size
            # U has shape = hidden_size x hidden_size
            # Product to be calculated is U x h
            # Hence take transpose of h_prev
            ufh = self.U_f @ torch.transpose(self.h_prev, 0, 1)
            # Take transpose of output to get
            # shape of single output array = 1 x hidden_size
            ufh = torch.transpose(ufh, 0, 1)

            # item has shape = 1 x embedding_size
            # W has shape = hidden_size x embedding_size
            # Product to be calculated is W x item
            # Hence take transpose of item
            wfx = self.W_f @ item.view(-1, 1)
            # Take transpose of output to get
            # shape of single output array = 1 x hidden_size
            wfx = torch.transpose(wfx, 0, 1)

            # Take the Sigmoid of the sum of these products
            f_t = self.sigmoid(ufh + wfx)

            # Remove information from context that is no longer required
            # by multiplying with context vector
            # Elementwise Multiplication for which both inputs
            # have shape 1 x hidden_size
            k_t = self.c_prev * f_t
            # k_t represents modified context vector

            # Compute actual information to be extracted from
            # the previous hidden state

            # h_prev has shape = 1 x hidden_size
            # U has shape = hidden_size x hidden_size
            # Product to be calculated is U x h
            # Hence take transpose of h_prev
            ugh = self.U_g @ torch.transpose(self.h_prev, 0, 1)
            # Take transpose of output to get
            # shape of single output array = 1 x hidden_size
            ugh = torch.transpose(ugh, 0, 1)

            # item has shape = 1 x embedding_size
            # W has shape = hidden_size x embedding_size
            # Product to be calculated is W x item
            # Hence take transpose of item
            wgx = self.W_g @ item.view(-1, 1)
            # Take transpose of output to get
            # shape of single output array = 1 x hidden_size
            wgx = torch.transpose(wgx, 0, 1)

            # Take Tanh of the sum of these products
            g_t = self.tanh(ugh + wgx)

            # Generate the mask for the Add Gate

            # h_prev has shape = 1 x hidden_size
            # U has shape = hidden_size x hidden_size
            # Product to be calculated is U x h
            # Hence take transpose of h_prev
            uih = self.U_i @ torch.transpose(self.h_prev, 0, 1)
            # Take transpose of output to get
            # shape of single output array = 1 x hidden_size
            uih = torch.transpose(uih, 0, 1)

            # item has shape = 1 x embedding_size
            # W has shape = hidden_size x embedding_size
            # Product to be calculated is W x item
            # Hence take transpose of item
            wix = self.W_i @ item.view(-1, 1)
            # Take transpose of output to get
            # shape of single output array = 1 x hidden_size
            wix = torch.transpose(wix, 0, 1)

            # Take Sigmoid of the sum of these products
            i_t = self.sigmoid(uih + wix)

            # Multiply with mask with actual information to select information
            # to add to the current context
            j_t = g_t * i_t

            # Add this to modified context vector (k_t)
            # to get new context vector
            c_t = j_t + k_t

            # generate mask for Output Gate

            # h_prev has shape = 1 x hidden_size
            # U has shape = hidden_size x hidden_size
            # Product to be calculated is U x h
            # Hence take transpose of h_prev
            uoh = self.U_o @ torch.transpose(self.h_prev, 0, 1)
            # Take transpose of output to get
            # shape of single output array = 1 x hidden_size
            uoh = torch.transpose(uoh, 0, 1)

            # item has shape = 1 x embedding_size
            # W has shape = hidden_size x embedding_size
            # Product to be calculated is W x item
            # Hence take transpose of item
            wox = self.W_o @ item.view(-1, 1)
            # Take transpose of output to get
            # shape of single output array = 1 x hidden_size
            wox = torch.transpose(wox, 0, 1)

            # Take Sigmoid of the sum
            o_t = self.sigmoid(uoh + wox)

            # generate final hidden state by taking elementwise product
            # of output gate mask and Tanh of the current context vector
            h_t = o_t * self.tanh(c_t)

            # Assign value to output
            h_out[i] = h_t

            # Designate current hidden state as previous
            self.h_prev = h_t

            # Designate current context vector as previous
            self.c_prev = c_t

        return h_out, c_out


class GRU(nn.Module):
    def __init__(self, embedding_size, hidden_size):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size

        # Initial Hidden State
        self.h_prev = torch.randn(1, self.hidden_size)

        # Weight Matrices for Reset Gate
        self.U_r = torch.randn(hidden_size, hidden_size)
        self.W_r = torch.randn(hidden_size, embedding_size)

        # Weight Matrices for Update Gate
        self.U_z = torch.randn(hidden_size, hidden_size)
        self.W_z = torch.randn(hidden_size, embedding_size)

        # Vector of Ones
        self.ones = torch.ones(1, self.hidden_size)

        # Sigmoid function
        self.sigmoid = nn.Sigmoid()

        # Weight Matrices for Intermediate Hidden State Representation
        self.U = torch.randn(hidden_size, hidden_size)
        self.W = torch.randn(hidden_size, embedding_size)

        # Tanh function
        self.tanh = nn.Tanh()

    def forward(self, inputs, h_0=None):
        # Assign custom initial hidden state if any
        if h_0 is not None:
            self.h_prev = h_0

        # Output tensor of hidden states
        # with shape (input_length x hidden_size)
        h_out = torch.zeros(inputs.size(-2), self.hidden_size)

        for i, item in enumerate(inputs):

            # Generate output of Reset Gate

            # h_prev has shape = 1 x hidden_size
            # U has shape = hidden_size x hidden_size
            # Product to be calculated is U x h
            # Hence take transpose of h_prev
            urh1 = self.U_r @ torch.transpose(self.h_prev, 0, 1)
            # Take transpose of output to get
            # shape of single output array = 1 x hidden_size
            urh1 = torch.transpose(urh1, 0, 1)

            # item has shape = 1 x embedding_size
            # W has shape = hidden_size x embedding_size
            # Product to be calculated is W x item
            # Hence take transpose of item
            wrx = self.W_r @ item.view(-1, 1)
            # Take transpose of output to get
            # shape of single output array = 1 x hidden_size
            wrx = torch.transpose(wrx, 0, 1)

            # Apply Sigmoid to sum of these products to get
            # output of reset gate
            r_t = self.sigmoid(urh1 + wrx)

            # Generate output of Update Gate

            # h_prev has shape = 1 x hidden_size
            # U has shape = hidden_size x hidden_size
            # Product to be calculated is U x h
            # Hence take transpose of h_prev
            uzh = self.U_z @ torch.transpose(self.h_prev, 0, 1)
            # Take transpose of output to get
            # shape of single output array = 1 x hidden_size
            uzh = torch.transpose(uzh, 0, 1)

            # item has shape = 1 x embedding_size
            # W has shape = hidden_size x embedding_size
            # Product to be calculated is W x item
            # Hence take transpose of item
            wzx = self.W_z @ item.view(-1, 1)
            # Take transpose of output to get
            # shape of single output array = 1 x hidden_size
            wzx = torch.transpose(wzx, 0, 1)

            # Apply Sigmoid to sum of these products to get
            # output of reset gate
            z_t = self.sigmoid(uzh + wzx)

            # Get Intermediate representation of Hidden State

            # Decide aspects of previous hidden state that are
            # relevant to current context
            rh = r_t * self.h_prev
            urh2 = self.U @ torch.transpose(rh, 0, 1)
            # Take transpose of output to get
            # shape of single output array = 1 x hidden_size
            urh2 = torch.transpose(urh2, 0, 1)

            # item has shape = 1 x embedding_size
            # W has shape = hidden_size x embedding_size
            # Product to be calculated is W x item
            # Hence take transpose of item
            wx = self.W @ item.view(-1, 1)
            # Take transpose of output to get
            # shape of single output array = 1 x hidden_size
            wx = torch.transpose(wx, 0, 1)

            # Take Tanh of the sum of these products
            h_intermediate = self.tanh(urh2 + wx)

            # Generate Final Hidden State
            h_t = (self.ones - z_t) * self.h_prev + z_t * h_intermediate

            # Assign value to output
            h_out[i] = h_t

            # Designate current hidden state as previous
            self.h_prev = h_t

        return h_out
