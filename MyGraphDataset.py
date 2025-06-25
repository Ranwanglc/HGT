import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.utils import from_networkx
import networkx as nx


class MyGraphDataset(Dataset):
    def __init__(self, data,num_classes,num_node_features, transform=None, pre_transform=None):
        """
        :param data_list: 包含多个 Data 对象的列表。
        :param transform: 用于数据的转换操作（例如数据增强）。
        :param pre_transform: 用于数据预处理操作（例如标准化）。
        """
        self.data = data  # 存储多个Data对象
        self.transform = transform
        self.pre_transform = pre_transform


    def len(self):
        """
        返回数据集的长度，即图的数量。
        """
        return len(self.data_list)

    def get(self):
        """
        获取指定索引的 Data 对象，并进行相应的转换。
        :param idx: 数据集的索引。
        :return: 对应的 Data 对象。
        """
        data = self.data
        if self.pre_transform:
            data = self.pre_transform(data)
        if self.transform:
            data = self.transform(data)
        return data

