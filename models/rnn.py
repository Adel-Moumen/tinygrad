"""Recurrent Neural Network (RNN) models."""

from tinygrad.tensor import Tensor
import numpy as np 


class RNN:
    def __init__(self, hidden_size, input_shape, num_layers=1, bias=True):
        self.hidden_size = hidden_size
        self.input_shape = input_shape 
        self.num_layers = num_layers
        self.bias = bias 

        self.batch_size = input_shape[0]
        self.fea_dim = input_shape[2]
        self.rnn = self._init_layers()

    def _init_layers(self):
        rnn = []
        current_dim = self.fea_dim

        for _ in range(self.num_layers):
            rnn_lay = RNNBlock(
                self.batch_size,
                current_dim,
                self.hidden_size,
                bias=self.bias
            )
            rnn.append(rnn_lay)
            current_dim = self.hidden_size
        return rnn 
        
    # TODO: make that hiddens is a Tensor not a list. 
    def __call__(self, x, hx=None):
        h = []
        for i, rnn_lay in enumerate(self.rnn):
            if hx is not None:
                x = rnn_lay(x, hx[i])
            else: 
                x = rnn_lay(x, None)
            h.append(x[-1])
        return x, h 

class RNNBlock:
    def __init__(self, batch_size, input_size, hidden_size, bias=True):
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size

        if bias:
            self.W_b = Tensor.ones(hidden_size)
            self.U_b = Tensor.ones(hidden_size)
        

        self.W = (Tensor.ones(hidden_size, input_size), self.W_b)
        self.U = (Tensor.ones(hidden_size, hidden_size), self.U_b)

        self.h_init = Tensor.zeros(self.batch_size, self.hidden_size)

    def __call__(self, x, hx=None):
        """input is (batch, time, features)"""

        if hx is not None:
            h = self._rnn_cell(x=x, ht=hx)
        else:
            h = self._rnn_cell(x=x, ht=self.h_init)
        return h 

    def _rnn_cell(self, x, ht):
        """"""
        hiddens = []
        wx = x.linear(self.W[0].transpose(order=(1, 0)), self.W[1])
        for t in range(x.shape[1]):
            ht = (wx[:, t] + ht.linear(self.U[0].transpose(order=(1, 0)), self.U[1])).relu()
            hiddens.append(ht)
        return hiddens

if __name__ == "__main__":

    x = Tensor.ones(1, 2, 2)
    rnn = RNN(2, x.shape, 2, True)

    x, h = rnn(x)

    


    