"""
Author: Minglang Yin, minglang_yin@brown.edu

Fast training version DeepONet
Only works for fix grid training data
"""
import torch
import torch.nn as nn

class opnn(nn.Module):
    def __init__(self, branch1_dim, branch2_dim, trunk_dim):
        super(opnn, self).__init__()
        # self.branch_dim = branch_dim
        # self.trunk_dim = trunk_dim
        self.z_dim = trunk_dim[-1]

        ## build branch net
        modules = []
        for i, h_dim in enumerate(branch1_dim):
            if i == 0:
                in_channels = h_dim
            else:
                modules.append(nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    nn.Tanh()
                    )
                )
                in_channels = h_dim
        self._branch1 = nn.Sequential(*modules)

        modules = []
        for i, h_dim in enumerate(branch2_dim):
            if i == 0:
                in_channels = h_dim
            else:
                modules.append(nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    nn.Tanh()
                    )
                )
                in_channels = h_dim
        self._branch2 = nn.Sequential(*modules)

        ## build trunk net
        modules = []
        for i, h_dim in enumerate(trunk_dim):
            if i == 0:
                in_channels = h_dim
            else:
                modules.append(nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    nn.Tanh()
                    )
                )
                in_channels = h_dim
        self._trunk = nn.Sequential(*modules)
        # self._out_layer = nn.Linear(self.z_dim, 1, bias = True)

    def forward(self, f, f_bc, x):
        """
        f: M*dim_f
        x: N*dim_x
        y_br: M*dim_h
        y_tr: N*dim_h
        y_out: y_br(ij) tensorprod y_tr(kj)
        """
        y_br1 = self._branch1(f)
        y_br2 = self._branch2(f_bc)
        y_br = y_br1*y_br2

        y_tr = self._trunk(x)
        y_out = torch.einsum("ij,kj->ik", y_br, y_tr)
        # print(y_out.shape)
        return y_out
    
    def loss(self, f, f_bc, x, y):
        y_out = self.forward(f, f_bc, x)
        loss = ((y_out - y)**2).mean()
        return loss