from turtledemo.penrose import start

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.nn import GCNConv
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
import numpy as np
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn import HGTConv, Sequential

from utils import cal_g_gradient3, cal_g_gradient1, cal_g_gradient2, cal_g_gradient4, cal_g_gradient5, \
    cal_g_gradient_gat
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils.hetero import construct_bipartite_edge_index
from typing import Optional, Tuple
from torch import Tensor
from torch_geometric.typing import Adj, OptTensor
from torch_sparse import SparseTensor, matmul

class GAT(nn.Module):
    def __init__(self, num_features, hidden_size, embedding_size, alpha):
        super(GAT, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.alpha = alpha
        self.conv1 = GATLayer(num_features, hidden_size, alpha)
        self.conv2 = GATLayer(hidden_size, embedding_size, alpha)

    def forward(self, x, adj, M):
        h = self.conv1(x, adj, M)
        h = self.conv2(h, adj, M)
        z = F.normalize(h, p=2, dim=1)
        A_pred = self.dot_product_decode(z)
        return A_pred, z

    def dot_product_decode(self, Z):
        A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
        return A_pred


class GATLayer(nn.Module):
    """
    Simple GAT layer
    """

    def __init__(self, in_features, out_features, alpha=0.2):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a_self = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a_self.data, gain=1.414)

        self.a_neighs = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a_neighs.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj, M, concat=True):
        h = torch.mm(input, self.W)

        attn_for_self = torch.mm(h, self.a_self)  # (N,1)
        attn_for_neighs = torch.mm(h, self.a_neighs)  # (N,1)
        attn_dense = attn_for_self + torch.transpose(attn_for_neighs, 0, 1)
        attn_dense = torch.mul(attn_dense, M)
        attn_dense = self.leakyrelu(attn_dense)  # (N,N)

        zero_vec = -9e15 * torch.ones_like(adj)
        adj = torch.where(adj > 0, attn_dense, zero_vec)
        attention = F.softmax(adj, dim=1)
        h_prime = torch.matmul(attention, h)

        if concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

class CustomHGTConv(HGTConv):
    def __init__(self, in_channels, out_channels, metadata, heads=1):
        super(CustomHGTConv, self).__init__(in_channels, out_channels, metadata, heads)
        self.attention_weights = None

    def forward(self, x_dict, edge_index_dict):
        out = super(CustomHGTConv, self).forward(x_dict, edge_index_dict)
        F = self.out_channels
        H = self.heads
        D = F // H

        k_dict, q_dict, v_dict, out_dict = {}, {}, {}, {}

        # Compute K, Q, V over node types:
        kqv_dict = self.kqv_lin(x_dict)
        for key, val in kqv_dict.items():
            k, q, v = torch.tensor_split(val, 3, dim=1)
            k_dict[key] = k.view(-1, H, D)
            q_dict[key] = q.view(-1, H, D)
            v_dict[key] = v.view(-1, H, D)

        q, dst_offset = self._cat(q_dict)
        k, v, src_offset = self._construct_src_node_feat(
            k_dict, v_dict, edge_index_dict)

        edge_index, edge_attr = construct_bipartite_edge_index(
            edge_index_dict, src_offset, dst_offset, edge_attr_dict=self.p_rel,
            num_nodes=k.size(0))
        self.attention_weights = edge_attr
        out = self.propagate(edge_index, k=k, q=q, v=v, edge_attr=edge_attr)

        # Reconstruct output node embeddings dict:
        for node_type, start_offset in dst_offset.items():
            end_offset = start_offset + q_dict[node_type].size(0)
            if node_type in self.dst_node_types:
                out_dict[node_type] = out[start_offset:end_offset]

        # Transform output node embeddings:
        a_dict = self.out_lin({
            k:
                torch.nn.functional.gelu(v) if v is not None else v
            for k, v in out_dict.items()
        })

        # Iterate over node types:
        for node_type, out in out_dict.items():
            out = a_dict[node_type]

            if out.size(-1) == x_dict[node_type].size(-1):
                alpha = self.skip[node_type].sigmoid()
                out = alpha * out + (1 - alpha) * x_dict[node_type]
            out_dict[node_type] = out

        return out_dict

class HGT(torch.nn.Module):
    def __init__(self,output_dim,data,xsize):
        super(HGT, self).__init__()
        self.conv1 = HGTConv(-1, 32, data.metadata(), heads=1)
        self.conv2 = HGTConv(-1, output_dim, data.metadata(), heads=1)
        self.decoder = torch.nn.ModuleDict({
            node_type: torch.nn.Linear(output_dim, data[node_type].x.size(1))
            for node_type in data.node_types
        })
        self.decoder00 = torch.nn.ModuleDict({
            node_type: torch.nn.Linear(output_dim, xsize)
            for node_type in data.node_types
        })
        self.attention_weights = []

    def forward(self, data):
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict
        x_dict = self.conv1(x_dict, edge_index_dict)
        # self.attention_weights.append(self.conv1.attention_weights)
        x_dict = self.conv2(x_dict, edge_index_dict)
        # self.attention_weights.append(self.conv2.attention_weights)

        # return {key: self.decoder00[key](x) for key, x in x_dict.items()},{key: self.decoder[key](x) for key, x in x_dict.items()}
        return {key: self.decoder[key](x) for key, x in x_dict.items()}

class SimpleGCN(torch.nn.Module):

    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self,args,otherEmbed,weight,dataset,data,num_classes,num_node_features,heterophily,cached: bool = False , bias: bool = False,add_self_loops: bool = True, node_dim: int = -2):
        super(SimpleGCN, self,).__init__()
        self.args = args
        self.conv1 = GCNConv(num_node_features, args.hidden)
        self.conv2 = GCNConv(args.hidden, data.x.size(1))
        self.conv3 = GCNConv(data.x.size(1), num_classes)
        self.heterophily = heterophily

        self.diffusion = args.diffusion
        self.isotherEmbeding = args.otherEmbeding
        # 尝试2参数
        self.lin1 = Linear(num_node_features, args.hidden, bias=False, weight_initializer='glorot')
        self.lin2 = Linear(args.hidden, data.x.size(1), bias=False, weight_initializer='glorot')
        self.lin3 = Linear(data.x.size(1), num_classes, bias=False, weight_initializer='glorot')
        self.cached = cached
        self._cached_edge_index = None
        self._cached_adj_t = None
        self.calg = 'cal_gradient_2'
        self.alpha = args.alpha
        self.beta = args.beta
        self.gamma = args.gamma
        self.sigma1 = args.sigma1
        self.sigma2 = args.sigma2
        self.add_self_loops = add_self_loops
        self.node_dim = np.int64(node_dim)
        self.k = args.k
        self.register_parameter('bias', None)
        #尝试3加入额外表示信息，可能是来自多样的表示信息
        self.otherEmbed = otherEmbed
        self.Weight = torch.nn.Parameter(weight)

        #尝试4加入MLP
        # self.num_mlp_layers = 2
        # self.mlp = Sequential('x, edge_index', [
        #     (torch.nn.Linear(dataset[0].x.size(1), dataset[0].x.size(1)), 'x -> x'),
        #     torch.nn.ReLU(inplace=True),
        #     *[
        #          (torch.nn.Linear(dataset[0].x.size(1), dataset[0].x.size(1)), 'x -> x'),
        #          torch.nn.ReLU(inplace=True)
        #      ] * (self.num_mlp_layers - 1),
        #     (torch.nn.Linear(dataset[0].x.size(1), dataset[0].x.size(1)), 'x -> x')
        # ])

    def reset_parameters(self):

        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, data ,edge_weight: OptTensor = None):
        # 尝试1，尝试将liner层改为conv层
        if self.args.conv:
            x, edge_index = data.x, data.edge_index
            x = self.conv1(x, edge_index)
            # x = self.lin1(x)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            x = self.conv2(x, edge_index)
            # x = self.lin2(x)

        else:
            x, edge_index = data.x, data.edge_index
            # x = self.conv1(x, edge_index)
            x = self.lin1(x)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            # x = self.conv2(x, edge_index)
            x = self.lin2(x)

        # 尝试3加入其他的表示信息--在diffusion前
        # if self.isotherEmbeding:
        #     x = self.heterophily * self.otherEmbed +(1-self.heterophily) * x

        # 尝试在之前加入MLP
        # x = self.mlp(F.relu(x), edge_index)
        # 尝试2加入扩散方程
        if self.diffusion:

            pygdata = data

            edgei = pygdata.edge_index
            # edgew =  self.Weight
            edgew = edge_weight
            cache = self._cached_edge_index
            if cache is None:
                edge_index, edge_weight = gcn_norm(  # yapf: disable
                    edgei, edgew, x.size(self.node_dim), False,
                    self.add_self_loops, dtype=x.dtype)
                edge_index2, edge_weight2 = gcn_norm(  # yapf: disable
                    edgei.type(torch.int64), edgew, x.size(self.node_dim), False,
                    False, dtype=x.dtype)

                if self.cached:
                    self._cached_edge_index = (edge_index, edge_weight)
            else:
                edge_index, edge_weight = cache[0], cache[1]
            ew = edge_weight.view(-1, 1)
            ew2 = edge_weight2.view(-1, 1)

            h = x
            for k in range(self.k):

                if self.calg == 'g3' or self.calg == 'cal_gradient_2':  # TODO
                    g = cal_g_gradient3(edge_index2, x, edge_weight=ew2, sigma1=self.sigma1, sigma2=self.sigma2)
                elif self.calg == 'g1':
                    g = cal_g_gradient1(edge_index2, x, edge_weight=ew2, sigma1=self.sigma1, sigma2=self.sigma2)
                elif self.calg == 'g2':
                    g = cal_g_gradient2(edge_index2, x, edge_weight=ew2, sigma1=self.sigma1, sigma2=self.sigma2)
                elif self.calg == 'g4':
                    g = cal_g_gradient4(edge_index2, x, edge_weight=ew2, sigma1=self.sigma1, sigma2=self.sigma2)
                elif self.calg == 'g5':
                    g = cal_g_gradient5(edge_index2, x, edge_weight=ew2, sigma1=self.sigma1, sigma2=self.sigma2)
                elif self.calg == 'ggat':
                    g = cal_g_gradient_gat(edge_index2, x, self.gat1, edge_weight=ew2, sigma1=self.sigma1,
                                           sigma2=self.sigma2)

                adj = torch.sparse_coo_tensor(edge_index, edge_weight, [x.size(0), x.size(0)])
                Ax = torch.spmm(adj, x)
                Gx = torch.spmm(adj, g)
                # if self.isotherEmbeding:
                #     x = self.alpha * h + (1 - self.alpha - self.beta) * x \
                #         + self.beta * torch.spmm(self.Matix,Ax) \
                #         + self.beta * self.gamma * torch.spmm(self.Matix,Gx)
                # else:
                x = self.alpha * h + (1 - self.alpha - self.beta) * x \
                    + self.beta * Ax \
                    + self.beta * self.gamma * Gx

        # 尝试3加入其他的表示信息
        if self.isotherEmbeding:
            x = self.heterophily * self.otherEmbed + (1-self.heterophily) * x
        #     x = torch.spmm(self.Matix,x)
        #     x = self.heterophily * self.otherEmbed + self.args.noheterophily * x
        # z = getOtherEmbeding(data,self.args)
        x = self.lin3(F.relu(x))
        # x = self.mlp(F.relu(x),edge_index)
        return F.log_softmax(x, dim=1)