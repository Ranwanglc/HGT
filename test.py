import torch
import os
import xlwt
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GCNConv
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn import MetaPath2Vec
from torch_geometric.datasets import IMDB
from torch_geometric.nn import HGTConv
from getArgs import get_args
from utils import cal_g_gradient3, cal_g_gradient1, cal_g_gradient2, cal_g_gradient4, cal_g_gradient5, \
    cal_g_gradient_gat
from torch_geometric.typing import Adj, OptTensor

args = get_args()
dataset = IMDB(root='...../IMDB')
data = dataset[0]

# 将数据和模型移动到GPU（如果可用）
# device = torch.device(f'{args.device}' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

data = data.to(device)

def getOtherEmbeding(hetero_data , args ,device):
    # 先进型数据集的转换，在进行嵌入表示
    # 初始化异质数据
    # hetero_data = HeteroData()
    #
    # # 假设根据节点标签将节点分为两类
    # for i, label in enumerate(data.y):
    #     node_type = f'class_{label.item()}'
    #     if node_type not in hetero_data.node_types:
    #         hetero_data[node_type].x = []
    #     hetero_data[node_type].x.append(data.x[i].unsqueeze(0))
    #
    # #手动设置一些元路径
    # metapath_list = []
    # # 连接节点并定义边类型
    # for src, dst in data.edge_index.t():
    #     src_type = f'class_{data.y[src].item()}'
    #     dst_type = f'class_{data.y[dst].item()}'
    #     edge_type = (src_type, 'to', dst_type)
    #
    #
    #     if edge_type not in hetero_data.edge_types:
    #         hetero_data[edge_type].edge_index = torch.empty((2, 0), dtype=torch.long)
    #
    #     hetero_data[edge_type].edge_index = torch.cat(
    #         [hetero_data[edge_type].edge_index, torch.tensor([[src], [dst]])], dim=1
    #     )
    #
    # # 将特征列表转换为张量
    # for node_type in hetero_data.node_types:
    #     hetero_data[node_type].x = torch.cat(hetero_data[node_type].x, dim=0)

    hetero_data.to(device)
    metapath_list = []
    # 使用metapath2vec进行嵌入表示
    if args.typeEmbeding == 'MetaPath2Vec':
        model = MetaPath2Vec(hetero_data.edge_index_dict, embedding_dim=128, metapath=metapath_list,
                             walk_length=5, context_size=3, walks_per_node=10,
                             num_negative_samples=1, sparse=True)

        # 将模型移动到GPU（如果可用）
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        # 优化器
        optimizer = torch.optim.SparseAdam(model.parameters(), lr=0.01)

        # 训练模型
        def train():
            model.train()
            total_loss = 0
            for _ in range(100):  # 迭代次数
                optimizer.zero_grad()
                loss = model.loss()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            return total_loss / 100

        # 开始训练
        for epoch in range(1, 101):
            loss = train()
            print(f'Epoch: {epoch}, Loss: {loss:.4f}')

        # 获取嵌入
        embeddings = model('author')
    elif args.typeEmbeding == 'HGT':
        model_HGT = HGT(hetero_data.num_features['actor'],hetero_data)
        embedings = model_HGT(hetero_data)
    else:
        raise ValueError(
            "do not content such type of embeding.")

    return embedings

class HGT(torch.nn.Module):
    def __init__(self,output_dim,data):
        super(HGT, self).__init__()
        self.conv1 = HGTConv(-1, 32, data.metadata(), heads=1)
        self.conv2 = HGTConv(-1, output_dim, data.metadata(), heads=1)

    def forward(self, data):
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict
        print('start')
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = self.conv2(x_dict, edge_index_dict)

        return x_dict





# la = torch.from_numpy(torch.load('...../project/weight/DAEGC/pretrain/label.pth'))
# print(la)
