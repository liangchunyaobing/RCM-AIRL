import torch
import torch.nn as nn
import torch.nn.functional as F


class ValueCNN(nn.Module):
    def __init__(self, path_feature, link_feature, input_dim, pad_idx=None):
        super(ValueCNN, self).__init__()
        self.path_feature = torch.from_numpy(path_feature).float()
        self.link_feature = torch.from_numpy(link_feature).float()
        self.pad_idx = pad_idx

        self.fc1 = nn.Linear(input_dim, 120)  # [batch, 120]
        self.fc2 = nn.Linear(120, 84)  # [batch, 84]
        self.fc3 = nn.Linear(84, 1)  # [batch, 8]

    def to_device(self, device):
        self.path_feature = self.path_feature.to(device)
        self.link_feature = self.link_feature.to(device)

    def process_features(self, state, des):
        # print('state', state.shape, 'des', des.shape)
        path_feature = self.path_feature[state, des, :]
        edge_feature = self.link_feature[state, :]
        feature = torch.cat([path_feature, edge_feature], -1)  # [batch_size, n_path_feature + n_edge_feature]
        return feature

    def forward(self, state, des):  # 这是policy
        x = self.process_features(state, des)
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = self.fc3(x)
        return x