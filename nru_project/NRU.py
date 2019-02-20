"""Implementation of the Non-Saturating Recurrent Unit(NRU)."""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class NRUCell(nn.Module):

    def __init__(self, device, input_size, hidden_size, memory_size=64, k=4,
                 activation="tanh", use_relu=False, layer_norm=False):
        super(NRUCell, self).__init__()

        self._device = device

        self.input_size = input_size
        self.hidden_size = hidden_size
        if activation == "tanh":
            self.activation = torch.tanh
        elif activation == "sigmoid":
            self.activation = torch.sigmoid

        self.memory_size = memory_size
        self.k = k
        self._use_relu =  use_relu
        self._layer_norm = layer_norm

        assert math.sqrt(self.memory_size*self.k).is_integer()
        sqrt_memk = int(math.sqrt(self.memory_size*self.k))
        self.hm2v_alpha = nn.Linear(self.memory_size + hidden_size, 2 * sqrt_memk)
        self.hm2v_beta = nn.Linear(self.memory_size + hidden_size, 2 * sqrt_memk)
        self.hm2alpha = nn.Linear(self.memory_size + hidden_size, self.k)
        self.hm2beta = nn.Linear(self.memory_size + hidden_size, self.k)

        if self._layer_norm:
            self._ln_h = nn.LayerNorm(hidden_size)

        self.hmi2h = nn.Linear(self.memory_size + hidden_size + self.input_size, hidden_size)

    def _opt_relu(self, x):
        if self._use_relu:
            return F.relu(x)
        else:
            return x

    def _opt_layernorm(self, x):
        if self._layer_norm:
            return self._ln_h(x)
        else:
            return x

    def forward(self, input, last_hidden):
        hidden = {}
        c_input = torch.cat((input, last_hidden["h"], last_hidden["memory"]), 1)

        h = F.relu(self._opt_layernorm(self.hmi2h(c_input)))

        # Flat memory equations
        alpha = self._opt_relu(self.hm2alpha(torch.cat((h,last_hidden["memory"]),1))).clone()
        beta = self._opt_relu(self.hm2beta(torch.cat((h,last_hidden["memory"]),1))).clone()

        u_alpha = self.hm2v_alpha(torch.cat((h,last_hidden["memory"]),1)).chunk(2,dim=1)
        v_alpha = torch.bmm(u_alpha[0].unsqueeze(2), u_alpha[1].unsqueeze(1)).view(-1, self.k, self.memory_size)
        v_alpha = self._opt_relu(v_alpha)
        v_alpha = torch.nn.functional.normalize(v_alpha, p=5, dim=2, eps=1e-12)
        add_memory = alpha.unsqueeze(2)*v_alpha

        u_beta = self.hm2v_beta(torch.cat((h,last_hidden["memory"]),1)).chunk(2, dim=1)
        v_beta = torch.bmm(u_beta[0].unsqueeze(2), u_beta[1].unsqueeze(1)).view(-1, self.k, self.memory_size)
        v_beta = self._opt_relu(v_beta)
        v_beta = torch.nn.functional.normalize(v_beta, p=5, dim=2, eps=1e-12)
        forget_memory = beta.unsqueeze(2)*v_beta

        hidden["memory"] = last_hidden["memory"] + torch.mean(add_memory-forget_memory, dim=1)
        hidden["h"] = h
        return hidden

    def reset_hidden(self, batch_size, hidden_init=None):
        hidden = {}
        if hidden_init is None:
            hidden["h"] = torch.Tensor(np.zeros((batch_size, self.hidden_size))).to(self._device)
        else:
            hidden["h"] = hidden_init.to(self._device)
        hidden["memory"] = torch.Tensor(np.zeros((batch_size, self.memory_size))).to(self._device)
        return hidden


class NRU(nn.Module):
    """Implementation of the Non-Saturating Recurrent Unit(NRU)."""

    def __init__(self, device, input_size, output_size, num_layers=1, layer_size=[10],
                 output_activation="linear", layer_norm=False, use_relu=False, memory_size=64, k=4):
        """Initialization parameters for NRU.
        
        Args:
            device: device_id ( GPU / CPU) to store the parameters.
            input_size: Number of expected features in the input.
            output_size: Number of expected features in the output.
            num_layers: Number of stacked recurrent layers.
            layer_size: The number of features in the hidden state
            output_activation: Activation function for the output layer.
            use_relu: If true, use ReLU activations over the erase/write memory vectors.
            memory_size: Number of dimensions of the memory vector.
            k: Number of erase/write heads.     
        
        """
        
        super(NRU, self).__init__()

        self._device = device
        self._input_size = input_size
        self._output_size = output_size
        self._num_layers = num_layers
        self._layer_size = layer_size
        self._output_activation = output_activation
        self._layer_norm = layer_norm
        self._use_relu = use_relu
        self._memory_size = memory_size
        self._k = k

        self._Cells = []

        self._add_cell(self._input_size, self._layer_size[0])
        for i in range(1, num_layers):
            self._add_cell(self._layer_size[i-1], self._layer_size[i])

        self._list_of_modules = nn.ModuleList(self._Cells)

        if self._output_activation == "linear":
            self._output_activation_fn = None
        elif self._output_activation == "sigmoid":
            self._output_activation_fn = F.sigmoid
        elif self._output_activation == "LogSoftmax":
            self._output_activation_fn = F.LogSoftmax

        self._W_h2o = nn.Parameter(torch.Tensor(layer_size[-1], output_size))
        self._b_o = nn.Parameter(torch.Tensor(output_size))

        self._reset_parameters()
        self.print_num_parameters()

    def forward(self, input, hidden_init=None):
        """Implements forward computation of the model.
        Input Args and outputs are in the same format as torch.nn.GRU cell.
        
        Args:
            input of shape (seq_len, batch, input_size): 
            tensor containing the features of the input sequence.
            Support for packed variable length sequence not available yet.
            
            hidden_init of shape (num_layers * num_directions, batch, hidden_size): 
            tensor containing the initial hidden state for each element in the batch. 
            Defaults to zero if not provided.
        
        Returns:
            output of shape (seq_len, batch, hidden_size): 
            tensor containing the output features h_t from the last layer of the GRU, for each t
            
            h_n of shape (num_layers, batch, hidden_size): 
            tensor containing the hidden state for t = seq_len 
        """
        batch_size = input.shape[1]
        seq_len = input.shape[0]

        hidden_inits = None
        if hidden_init is not None:                                  
            hidden_inits = torch.chunk(hidden_init, len(self._Cells), dim=0)

        self.reset_hidden(batch_size, hidden_inits)

        outputs = []
        for t in range(seq_len):
            outputs.append(self.step(input[t]))
        
        #output contruction
        output = torch.stack([h[-1] for h in outputs], dim=0)

        #h_n construction
        h_n = torch.stack([hidden["h"] for hidden in self._h_prev], dim=0)
        
        return output, h_n

    def step(self, input):
        """Implements forward computation of the model for a single recurrent step.

        Args:
            input of shape (batch, input_size): 
            tensor containing the features of the input sequence

        Returns:
            model output for current time step.
        """
        
        h = []
        h.append(self._Cells[0](input, self._h_prev[0]))
        for i, cell in enumerate(self._Cells):
            if i != 0:
                h.append(cell(h[i-1]["h"], self._h_prev[i]))
        output = torch.add(torch.mm(h[-1]["h"], self._W_h2o), self._b_o)
        if self._output_activation_fn is not None:
            output = self._output_activation_fn(output)
        self._h_prev = h
        return output

    def reset_hidden(self, batch_size, hidden_inits=None):
        """Resets the hidden state for truncating the dependency."""

        self._h_prev = []
        for i, cell in enumerate(self._Cells):
            self._h_prev.append(cell.reset_hidden(batch_size, hidden_inits[i] if hidden_inits is not None else None))

    def _reset_parameters(self):
        """Initializes the parameters."""

        nn.init.xavier_normal_(self._W_h2o, gain=nn.init.calculate_gain(self._output_activation))
        nn.init.constant_(self._b_o, 0)

    def register_optimizer(self, optimizer):
        """Registers an optimizer for the model.

        Args:
            optimizer: optimizer object.
        """

        self.optimizer = optimizer

    def _add_cell(self, input_size, hidden_size):
        """Adds a cell to the stack of cells.

        Args:
            input_size: int, size of the input vector.
            hidden_size: int, hidden layer dimension.
        """

        self._Cells.append(NRUCell(self._device, input_size, hidden_size, 
                                              memory_size=self._memory_size, k=self._k,
                                              use_relu=self._use_relu, layer_norm=self._layer_norm))
                                    

    def save(self, save_dir):
        """Saves the model and the optimizer.

        Args:
            save_dir: absolute path to saving dir.
        """

        file_name = os.path.join(save_dir, "model.p")
        torch.save(self.state_dict(), file_name)

        file_name = os.path.join(save_dir, "optim.p")
        torch.save(self.optimizer.state_dict(), file_name)

    def load(self, save_dir):
        """Loads the model and the optimizer.

        Args:
            save_dir: absolute path to loading dir.
        """

        file_name = os.path.join(save_dir, "model.p")
        self.load_state_dict(torch.load(file_name))

        file_name = os.path.join(save_dir, "optim.p")
        self.optimizer.load_state_dict(torch.load(file_name))

    def print_num_parameters(self):
        num_params = 0
        for p in self.parameters():
            num_params += p.data.view(-1).size(0)
        print("Num_params : {} ".format(num_params))
        return num_params
