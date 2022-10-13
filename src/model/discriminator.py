import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DiscriminatorAIRLCNN(nn.Module):
    def __init__(self, action_num, gamma, policy_mask, action_state, path_feature, link_feature, rs_input_dim,
                 hs_input_dim, pad_idx=None):
        super(DiscriminatorAIRLCNN, self).__init__()
        self.gamma = gamma
        self.policy_mask = torch.from_numpy(policy_mask).long()
        policy_mask_pad = np.concatenate([policy_mask, np.zeros((policy_mask.shape[0], 1), dtype=np.int32)], 1)
        self.policy_mask_pad = torch.from_numpy(policy_mask_pad).long()
        action_state_pad = np.concatenate([action_state, np.expand_dims(np.arange(action_state.shape[0]), 1)], 1)
        self.action_state_pad = torch.from_numpy(action_state_pad).long()
        self.path_feature = torch.from_numpy(path_feature).float()
        self.link_feature = torch.from_numpy(link_feature).float()
        self.new_index = torch.tensor([7, 0, 1, 6, 8, 2, 5, 4, 3]).long()
        self.pad_idx = pad_idx
        self.action_num = action_num

        self.conv1 = nn.Conv2d(rs_input_dim, 20, 3, padding=1)  # [batch, 20, 3, 3]
        self.pool = nn.MaxPool2d(2, 1)  # [batch, 20, 3, 3]
        self.conv2 = nn.Conv2d(20, 30, 2)  # [batch, 30, 1, 1]
        self.fc1 = nn.Linear(30 + self.action_num, 120)  # [batch, 120]
        self.fc2 = nn.Linear(120, 84)  # [batch, 84]
        self.fc3 = nn.Linear(84, 1)  # [batch, 8]

        self.h_fc1 = nn.Linear(hs_input_dim, 120)  # [batch, 120]
        self.h_fc2 = nn.Linear(120, 84)  # [batch, 84]
        self.h_fc3 = nn.Linear(84, 1)  # [batch, 8]

    def to_device(self, device):
        self.policy_mask = self.policy_mask.to(device)
        self.policy_mask_pad = self.policy_mask_pad.to(device)
        self.action_state_pad = self.action_state_pad.to(device)
        self.path_feature = self.path_feature.to(device)
        self.link_feature = self.link_feature.to(device)
        self.new_index = self.new_index.to(device)

    def process_neigh_features(self, state, des):
        state_neighbor = self.action_state_pad[state]
        neigh_path_feature = self.path_feature[state_neighbor, des.unsqueeze(1).repeat(1, self.action_num + 1),
                             :]
        neigh_edge_feature = self.link_feature[state_neighbor, :]
        neigh_mask_feature = self.policy_mask_pad[state].unsqueeze(-1)  # [batch_size, 9, 1]
        neigh_feature = torch.cat([neigh_path_feature, neigh_edge_feature, neigh_mask_feature],
                                  -1)
        neigh_feature = neigh_feature[:, self.new_index, :]
        x = neigh_feature.view(state.size(0), 3, 3, -1)
        x = x.permute(0, 3, 1, 2)
        return x

    def process_state_features(self, state, des):
        path_feature = self.path_feature[state, des, :]  # 实在不行你也可以把第一个dimension拉平然后reshape 一下
        edge_feature = self.link_feature[state, :]
        feature = torch.cat([path_feature, edge_feature], -1)  # [batch_size, n_path_feature + n_edge_feature]
        return feature

    def f(self, state, des, act, next_state):
        """rs"""
        x = self.process_neigh_features(state, des)
        x = self.pool(F.leaky_relu(self.conv1(x), 0.2))
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = x.view(-1, 30)  # 到这一步等于是对这个3x3的图提取feature
        x_act = F.one_hot(act, num_classes=self.action_num)
        x = torch.cat([x, x_act], 1)  # [batch_size, 38]
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)  # 我个人的建议是你先把它按照图像处理完
        rs = self.fc3(x)

        """hs"""
        x_state = self.process_state_features(state, des)
        x_state = F.leaky_relu(self.h_fc1(x_state), 0.2)
        x_state = F.leaky_relu(self.h_fc2(x_state), 0.2)
        x_state = self.h_fc3(x_state)

        """hs_next"""
        next_x_state = self.process_state_features(next_state, des)
        next_x_state = F.leaky_relu(self.h_fc1(next_x_state), 0.2)
        next_x_state = F.leaky_relu(self.h_fc2(next_x_state), 0.2)
        next_x_state = self.h_fc3(next_x_state)

        return rs + self.gamma * next_x_state - x_state

    def forward(self, states, des, act, log_pis, next_states):
        # Discriminator's output is sigmoid(f - log_pi).
        return self.f(states, des, act, next_states) - log_pis

    def calculate_reward(self, states, des, act, log_pis, next_states):
        with torch.no_grad():
            logits = self.forward(states, des, act, log_pis, next_states)
            return -F.logsigmoid(-logits)


class DiscriminatorCNN(nn.Module):
    def __init__(self, action_num, policy_mask, action_state, path_feature, link_feature, input_dim, pad_idx=None):
        super(DiscriminatorCNN, self).__init__()
        self.policy_mask = torch.from_numpy(policy_mask).long()
        policy_mask_pad = np.concatenate([policy_mask, np.zeros((policy_mask.shape[0], 1), dtype=np.int32)], 1)
        self.policy_mask_pad = torch.from_numpy(policy_mask_pad).long()
        action_state_pad = np.concatenate([action_state, np.expand_dims(np.arange(action_state.shape[0]), 1)], 1)
        self.action_state_pad = torch.from_numpy(action_state_pad).long()
        self.path_feature = torch.from_numpy(path_feature).float()
        self.link_feature = torch.from_numpy(link_feature).float()
        self.new_index = torch.tensor([7, 0, 1, 6, 8, 2, 5, 4, 3]).long()
        self.pad_idx = pad_idx
        self.action_num = action_num

        self.conv1 = nn.Conv2d(input_dim, 20, 3, padding=1)  # [batch, 20, 3, 3]
        self.pool = nn.MaxPool2d(2, 1)  # [batch, 20, 3, 3]
        self.conv2 = nn.Conv2d(20, 30, 2)  # [batch, 30, 1, 1]
        self.fc1 = nn.Linear(30 + self.action_num, 120)  # [batch, 120]
        self.fc2 = nn.Linear(120, 84)  # [batch, 84]
        self.fc3 = nn.Linear(84, 1)  # [batch, 8]

    def to_device(self, device):
        self.policy_mask = self.policy_mask.to(device)
        self.policy_mask_pad = self.policy_mask_pad.to(device)
        self.action_state_pad = self.action_state_pad.to(device)
        self.path_feature = self.path_feature.to(device)
        self.link_feature = self.link_feature.to(device)
        self.new_index = self.new_index.to(device)

    def process_features(self, state, des):
        state_neighbor = self.action_state_pad[state]
        neigh_path_feature = self.path_feature[state_neighbor, des.unsqueeze(1).repeat(1, self.action_num + 1), :]
        neigh_edge_feature = self.link_feature[state_neighbor, :]
        neigh_mask_feature = self.policy_mask_pad[state].unsqueeze(-1)  # [batch_size, 9, 1]
        neigh_feature = torch.cat([neigh_path_feature, neigh_edge_feature, neigh_mask_feature],
                                  -1)  # [batch_size, 9, n_path_feature + n_edge_feature + 1]
        neigh_feature = neigh_feature[:, self.new_index, :]
        x = neigh_feature.view(state.size(0), 3, 3, -1)
        x = x.permute(0, 3, 1, 2)
        # print('x', x.shape)
        return x

    def forward(self, state, des, act):  # 这是policy
        x = self.process_features(state, des)
        x = self.pool(F.leaky_relu(self.conv1(x), 0.2))
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = x.view(-1, 30)  # 到这一步等于是对这个3x3的图提取feature

        x_act = F.one_hot(act, num_classes=self.action_num)

        x = torch.cat([x, x_act], 1)  # [batch_size, 38]
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)  # 我个人的建议是你先把它按照图像处理完

        prob = torch.sigmoid(self.fc3(x))
        return prob

    def calculate_reward(self, st, des, act):
        # PPO(GAIL) is to maximize E_{\pi} [-log(1 - D)].
        with torch.no_grad():
            return -torch.log(self.forward(st, des, act))