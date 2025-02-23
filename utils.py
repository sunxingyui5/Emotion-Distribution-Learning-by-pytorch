import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


class GetData(Dataset):
    def __init__(self, path: str):  # 得到名字list
        """
        加载数据集
        :param path: 加载文件，包括 data 和 target
        """
        super(GetData, self).__init__()
        data = torch.load(path)
        self.data = data["data"]
        self.target = data["target"]

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        return self.data[index], self.target[index]


# 切比雪夫距离
def chebyshev_distance(x, y):
    mx, _ = torch.max(abs(x - y), 1)
    return torch.sum(mx)


# 堪培拉距离
def canberra_distance(x, y):
    return torch.sum(abs(x - y) / (abs(x) + abs(y)))


# 余弦相似度
def cosine_similarity(x, y):
    return torch.sum(F.cosine_similarity(x, y, 1))


# 准确度
def acc(x, y):
    return torch.sum(torch.argmax(x, dim=1) == torch.argmax(y, dim=1))


if __name__ == '__main__':
    data = GetData("Flickr_LDL/test.tf")
    print(data[0][0].shape, data[0][1].shape)
