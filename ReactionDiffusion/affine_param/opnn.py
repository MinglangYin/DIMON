"""
Author: Minglang Yin, myin16@jhu.edu
"""
import torch
import torch.nn as nn

class opnn(nn.Module):
    def __init__(self, branch_u_dim, branch_g_dim, trunk_dim, num_snap, num_pts):
        super(opnn, self).__init__()
        # self.branch_u_dim = branch_u_dim
        # self.branch_r_dim = branch_r_dim
        # self.merge_dim = merge_dim
        # self.z_dim = trunk_dim[-1]

        self.num_snap = num_snap
        self.num_pts = num_pts

        ## build branch1 net
        modules = []
        for i, h_dim in enumerate(branch_u_dim):
            if i == 0:
                in_channels = h_dim
            else:
                modules.append(nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    nn.Tanh()
                    # nn.ReLU()
                    )
                )
                in_channels = h_dim
        self._branch_u = nn.Sequential(*modules)

        ## build branch3 net
        modules = []
        for i, h_dim in enumerate(branch_g_dim):
            if i == 0:
                in_channels = h_dim
            else:
                modules.append(nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    # nn.ReLU()
                    nn.Tanh()
                    )
                )
                in_channels = h_dim
        self._branch_g = nn.Sequential(*modules)

        ## build trunk net
        modules = []
        for i, h_dim in enumerate(trunk_dim):
            if i == 0:
                in_channels = h_dim
            else:
                modules.append(nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    nn.Tanh()
                    # nn.ReLU()
                    )
                )
                in_channels = h_dim
        self._trunk1 = nn.Sequential(*modules)
        self._out_layer1 = nn.Linear(in_channels, in_channels, bias = True)

        # modules = []
        # for i, h_dim in enumerate(trunk_dim):
        #     if i == 0:
        #         in_channels = h_dim
        #     else:
        #         modules.append(nn.Sequential(
        #             nn.Linear(in_channels, h_dim),
        #             nn.Tanh()
        #             )
        #         )
        #         in_channels = h_dim
        # self._trunk2 = nn.Sequential(*modules)
        # # self._out_layer2 = nn.Linear(in_channels, in_channels, bias = True)

        # modules = []
        # for i, h_dim in enumerate(trunk_dim):
        #     if i == 0:
        #         in_channels = h_dim
        #     else:
        #         modules.append(nn.Sequential(
        #             nn.Linear(in_channels, h_dim),
        #             nn.Tanh()
        #             )
        #         )
        #         in_channels = h_dim
        # self._trunk3 = nn.Sequential(*modules)
        # # self._out_layer3 = nn.Linear(in_channels, in_channels, bias = True)

        # modules = []
        # for i, h_dim in enumerate(trunk_dim):
        #     if i == 0:
        #         in_channels = h_dim
        #     else:
        #         modules.append(nn.Sequential(
        #             nn.Linear(in_channels, h_dim),
        #             nn.Tanh()
        #             )
        #         )
        #         in_channels = h_dim
        # self._trunk4 = nn.Sequential(*modules)
        # self._out_layer4 = nn.Linear(in_channels, in_channels, bias = True)

        self._out_layer = nn.Linear(in_channels, in_channels, bias = True)


    def forward(self, f_u, f_g, x):
        """
        f: M*dim_f
        x: N*dim_x
        y_br: M*dim_h
        y_tr: N*dim_h
        y_out: y_br(ij) tensorprod y_tr(kj)
        """

        y_br_u = self._branch_u(f_u)
        y_br_g = self._branch_g(f_g) ## [num_case, dim_latent]
        # y_ = self._out_layer1(y_br_u*y_br_g)
        y_ = y_br_u*y_br_g
        # print(y_.shape)

        # y_merge_out = self._merge(y_)

        # y_tr1 = self._out_layer1(self._trunk1(x))
        # y_tr2 = self._out_layer2(self._trunk2(2*x))
        # y_tr3 = self._out_layer3(self._trunk3(4*x))
        # y_tr4 = self._out_layer4(self._trunk4(8*x))
        # y_tr = y_tr1*y_tr2*y_tr3*y_tr4
        
        # y_tr = self._trunk1(x)
        y_tr = self._trunk1(x)
        # y_tr2 = self._trunk2(2*x)
        # y_tr3 = self._trunk3(4*x)
        # y_tr4 = self._trunk4(8*x)
        # y_tr = self._out_layer(y_tr1*y_tr2*y_tr3*y_tr4)

        # print(y_tr.shape)

        # print(y_.shape, y_tr.shape)
        # exit()

        y_out = torch.einsum("ij, kj->ik", y_, y_tr)

        # print(y_out.shape)

        # y_out = torch.einsum("ijk,lk->ijl", y_, y_tr)

        return y_out.reshape(-1, self.num_snap, self.num_pts)
