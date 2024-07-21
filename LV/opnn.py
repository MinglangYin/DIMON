"""
Author: Minglang Yin, myin16@jhu.edu
"""
import torch
import torch.nn as nn

class opnn(nn.Module):
    def __init__(self, branch_g_dim, branch_p_dim, trunk_dim):
        super(opnn, self).__init__()

        ## build branch1 net
        modules = []
        for i, h_dim in enumerate(branch_g_dim):
            if i == 0:
                in_channels = h_dim
            else:
                modules.append(nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    nn.Tanh()
                    )
                )
                in_channels = h_dim
        self._branch_g = nn.Sequential(*modules)

        ## build branch net pace
        modules = []
        for i, h_dim in enumerate(branch_p_dim):
            if i == 0:
                in_channels = h_dim
            else:
                modules.append(nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    nn.Tanh()
                    )
                )
                in_channels = h_dim
        self._branch_p = nn.Sequential(*modules)

        # ## build merge net
        # modules = []
        # for i, h_dim in enumerate(merge_dim):
        #     if i == 0:
        #         in_channels = h_dim
        #     else:
        #         modules.append(nn.Sequential(
        #             nn.Linear(in_channels, h_dim),
        #             nn.Tanh()
        #             )
        #         )
        #         in_channels = h_dim
        # modules.append(nn.Sequential(nn.Linear(in_channels, h_dim)))
        # self._merge = nn.Sequential(*modules)
        # self._out_layer = nn.Linear(self.z_dim, 1, bias = True)

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
        # self._out_layer = nn.Linear(in_channels, in_channels, bias = True)


    def forward(self, f_g, x_pace, x):
        """
        f: M*dim_f
        x: N*dim_x
        y_br: M*dim_h
        y_tr: N*dim_h
        y_out: y_br(ij) tensorprod y_tr(kj)
        """

        y_br_g = self._branch_g(f_g)
        y_br_p = self._branch_p(x_pace) ## [num_case, dim_latent]
        y_ = y_br_p * y_br_g[:, None]
        # print(y_.shape)
        # exit()

        # print(f_g.shape, x_pace.shape, y_br_g.shape, y_br_p.shape, y_.shape)
        # exit()
        y_tr = self._trunk(x)
        y_out = torch.einsum("ijk,lk->ijl", y_, y_tr)
        return y_out
