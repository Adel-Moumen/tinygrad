"""Recurrent Neural Network (RNN) models."""

from tinygrad.tensor import Tensor

class RNNBlock:
    def __init__(self, batch_size, input_size, hidden_size, bias=True):
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size

        if bias:
            self.W_b = Tensor.uniform(hidden_size)
            self.U_b = Tensor.uniform(hidden_size)
        else:
            self.W_b = Tensor.zeros(hidden_size)
            self.U_b = Tensor.zeros(hidden_size)


        self.W = (Tensor.uniform(input_size, hidden_size), self.W_b)
        self.U = (Tensor.uniform(hidden_size, hidden_size), self.U_b)

        self.h_init = Tensor.zeros(self.batch_size, self.hidden_size)

    def __call__(self, x, hx=None):
        """input is (batch, time, features)"""

        if hx is not None:
            h = self._rnn_cell(x=x, hx=hx)
        else:
            h = self._rnn_cell(x=x, hx=self.h_init)
        
        return h 

    def _rnn_cell(self, x, ht):

        wx = x.linear(*self.W)
        for t in range(x.shape[0]):
            ht = wx + ht.linear(*self.U)
        
        return ht 

if __name__ == "__main__":
    rnn = RNNBlock(1, 1, 2)

    x = Tensor.randn(1, 1)