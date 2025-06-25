
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import HeteroData
from torch_geometric.nn import MetaPath2Vec
import numpy as np

from models import HGT,GCN
import torch.nn.functional as F
from sklearn.cluster import KMeans, DBSCAN
from daegc import getL
from pretrain import getPreL
def getDAEGC_label(args):
    getPreL(args)
    label = getL(args)
    return label

def getOtherEmbeding(dataset,data , args):
    # 方法1：先使用GCN+K_m进行聚类，再用聚类结果进行异质图的回归分析。
    label_julei = getDAEGC_label(args)
    label_julei = torch.from_numpy(label_julei)
    # 方法2：先进型数据集的转换，在进行嵌入表示
    # 初始化异质数据
    hetero_data = HeteroData()
    down_lable = np.zeros(len(data.y))
    x_lableName = []
    # 假设根据节点标签将节点分为两类
    for i, label in enumerate(label_julei):
        node_type = f'class_{label.item()}'
        if node_type not in hetero_data.node_types:
            hetero_data[node_type].x = []
        down_lable[i] = len(hetero_data[node_type].x)
        hetero_data[node_type].x.append(data.x[i].unsqueeze(0))
        x_lableName.append(node_type)


    #手动设置一些元路径
    metapath_list = []
    # 连接节点并定义边类型
    for src, dst in data.edge_index.t():
        src_type = f'class_{label_julei[src].item()}'
        dst_type = f'class_{label_julei[dst].item()}'
        edge_type = (src_type, 'to', dst_type)


        if edge_type not in hetero_data.edge_types:
            hetero_data[edge_type].edge_index = torch.empty((2, 0), dtype=torch.long)

        hetero_data[edge_type].edge_index = torch.cat(
            [hetero_data[edge_type].edge_index, torch.tensor([[int(down_lable[src])], [int(down_lable[dst])]])], dim=1
        )

    # 将特征列表转换为张量
    for node_type in hetero_data.node_types:
        hetero_data[node_type].x = torch.cat(hetero_data[node_type].x, dim=0)

    hetero_data = hetero_data.to(args.device)

    # embeddings = torch.zeros(data.x.size(0), data.x.size(0), dtype=torch.float).to(args.device)
    embeddings = torch.zeros(data.x.size(0), data.x.size(1), dtype=torch.float).to(args.device)
    auto_embeddings = torch.zeros(data.x.size(0), data.x.size(0), dtype=torch.float).to(args.device)
    # 使用metapath2vec进行嵌入表示
    # if args.typeEmbeding == 'MetaPath2Vec':
    #     model = MetaPath2Vec(hetero_data.edge_index_dict, embedding_dim=128, metapath=metapath_list,
    #                          walk_length=5, context_size=3, walks_per_node=10,
    #                          num_negative_samples=1, sparse=True)
    #
    #     # 将模型移动到GPU（如果可用）
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     model = model.to(device)
    #
    #     # 优化器
    #     optimizer = torch.optim.SparseAdam(model.parameters(), lr=0.01)
    #
    #     # 训练模型
    #     def train():
    #         model.train()
    #         total_loss = 0
    #         for _ in range(100):  # 迭代次数
    #             optimizer.zero_grad()
    #             loss = model.loss()
    #             loss.backward()
    #             optimizer.step()
    #             total_loss += loss.item()
    #         return total_loss / 100
    #
    #     # 开始训练
    #     for epoch in range(1, 101):
    #         loss = train()
    #         print(f'Epoch: {epoch}, Loss: {loss:.4f}')
    #
    #     # 获取嵌入
    #     return  model('author')
    if args.typeEmbeding == 'HGT':
        model_HGT = HGT(64,hetero_data,data.x.size(0)).to(args.device)
        embeding_temp = model_HGT(hetero_data)
        optimizer = torch.optim.Adam(model_HGT.parameters(), lr=0.01, weight_decay=5e-4)
        y = label_julei.float()

        train_mask, test_mask, = train_test_split(range(y.size(0)), test_size=0.2, random_state=42)

        # 训练模型
        def train():
            # embeddings_temp01 = torch.zeros(data.x.size(0), dataset.num_classes, dtype=torch.float).to(args.device)
            model_HGT.train()
            optimizer.zero_grad()
            out = model_HGT(hetero_data)
            # 以shiyong mse_loss
            # for i in range(len(data.y)):
            #    embeddings_temp01[i][:] = out[x_lableName[i]][int(down_lable[i])]
            # embeddings_temp01 = F.log_softmax(embeddings_temp01, dim=1)
            # loss = F.mse_loss(embeddings_temp01[train_mask].squeeze(), y[train_mask])
            loss = 0
            for node_type in hetero_data.node_types:
                loss += F.mse_loss(out[node_type], hetero_data[node_type].x)
            loss.backward()
            optimizer.step()
            return loss.item()

        # 测试模型
        # def test():
        #     embeddings_temp01 = torch.zeros(data.x.size(0), dataset.num_classes, dtype=torch.float).to(args.device)
        #     model_HGT.eval()
        #     with torch.no_grad():
        #         out = model_HGT(hetero_data)
        #         for i in range(len(data.y)):
        #             embeddings_temp01[i][:] = out[x_lableName[i]][int(down_lable[i])]
        #         pred = embeddings_temp01.argmax(dim=1)
        #         correct = (pred[test_mask] == y[test_mask]).sum()
        #         acc = int(correct) / len(test_mask)
        #     return acc
        for epoch in range(1, 101):
            loss = train()
            # acc = test()
            embeding_temp  = model_HGT(hetero_data)
            # print(f'Epoch: {epoch}, Loss: {loss:.4f}, Test Acc: {acc:.4f}')
            # print(f'Epoch: {epoch}, Loss: {loss:.4f}')

        weight = model_HGT.attention_weights
        for i in range(len(data.y)):
            embeddings[i][:] = embeding_temp[x_lableName[i]][int(down_lable[i])]
            # auto_embeddings[i][:] = embeding_temp1[x_lableName[i]][int(down_lable[i])]

    else:
        raise ValueError(
            "do not content such type of embeding.")
    # ,weight[len(weight)-1]
    if args.dataset == 'chameleon':
        torch.save(embeddings, '...../project/weight/DAEGC/pretrain/embeddings.pth')
    else:
        torch.save(embeddings, '...../project/weight/DAEGC/pretrain/'+args.dataset+'embeddings.pth')
    return embeddings





