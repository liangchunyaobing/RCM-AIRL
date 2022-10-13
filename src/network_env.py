from collections import namedtuple
import torch
import numpy as np
import pandas as pd

Step = namedtuple('Step', ['cur_state', 'action', 'next_state', 'reward', 'done'])


class RoadWorld(object):
    """
    Environment
    """

    def __init__(self, network_path, edge_path, pre_reset=None, origins=None,
                 destinations=None, k=8):
        self.network_path = network_path
        self.netin = origins
        self.netout = destinations
        self.k = k
        self.max_route_length = 0

        # define transition matrix
        netconfig = np.load(self.network_path)
        netconfig = pd.DataFrame(netconfig, columns=["from", "con", "to"])
        netconfig_dict = {}
        for i in range(len(netconfig)):
            fromid, con, toid = netconfig.loc[i]

            if fromid in netconfig_dict.keys():
                netconfig_dict[fromid][con] = toid
            else:
                netconfig_dict[fromid] = {}
                netconfig_dict[fromid][con] = toid
        self.netconfig = netconfig_dict

        # define states and actions
        edge_df = pd.read_csv(edge_path, header=0, usecols=['n_id'])

        # self.terminal = len(self.states)  # add a terminal state for destination
        self.states = edge_df['n_id'].tolist()
        self.pad_idx = len(self.states)
        self.states.append(self.pad_idx)
        self.actions = range(k)  # k represents action to terminal

        self.n_states = len(self.states)
        self.n_actions = len(self.actions)
        print('n_states', self.n_states)
        print('n_actions', self.n_actions)

        self.rewards = [0 for _ in range(self.n_states)]

        self.state_action_pair = sum([[(s, a) for a in self.get_action_list(s)] for s in self.states], [])
        self.num_sapair = len(self.state_action_pair)
        print('n_sapair', self.num_sapair)

        self.sapair_idxs = self.state_action_pair  # I think in our case this two should be the same
        self.policy_mask = np.zeros([self.n_states, self.n_actions], dtype=np.int32)
        self.state_action = np.ones([self.n_states, self.n_actions], dtype=np.int32) * self.pad_idx
        print('policy mask', self.policy_mask.shape)
        for s, a in self.sapair_idxs:
            self.policy_mask[s, a] = 1
            self.state_action[s, a] = self.netconfig[s][a]

        self.cur_state = None
        self.cur_des = None

        if pre_reset is not None:
            self.od_list = pre_reset[0]
            self.od_dist = pre_reset[1]

    def reset(self, st=None, des=None):
        if st is not None and des is not None:
            self.cur_state, self.cur_des = st, des
        else:
            od_idx = np.random.choice(self.od_list, 1, p=self.od_dist)
            ori, des = od_idx[0].split('_')
            self.cur_state, self.cur_des = int(ori), int(des)
        return self.cur_state, self.cur_des

    def get_reward(self, state):
        return self.rewards[state]

    def step(self, action):
        """
        Step function for the agent to interact with gridworld
        inputs:
          action        action taken by the agent
        returns
          current_state current state
          action        input action
          next_state    next_state
          reward        reward on the next state
          is_done       True/False - if the agent is already on the terminal states
        """
        tmp_dict = self.netconfig.get(self.cur_state, None)
        if tmp_dict is not None:
            next_state = tmp_dict.get(action, self.pad_idx)
        else:
            next_state = self.pad_idx
        reward = self.get_reward(self.cur_state)
        self.cur_state = next_state
        done = (self.cur_state == self.cur_des) or (self.cur_state == self.pad_idx)
        return next_state, reward, done

    def get_state_transition(self, state, action):
        return self.netconfig[state][action]

    def get_action_list(self, state):
        if state in self.netconfig.keys():
            return list(self.netconfig[state].keys())
        else:
            return list()

    def import_demonstrations(self, demopath, od=None, n_rows=None):
        demo = pd.read_csv(demopath, header=0, nrows=n_rows)
        expert_st, expert_des, expert_ac, expert_st_next = [], [], [], []
        for demo_str, demo_des in zip(demo['path'].tolist(), demo['des'].tolist()):
            cur_demo = [int(r) for r in demo_str.split('_')]
            len_demo = len(cur_demo)
            for i0 in range(1, len_demo):
                cur_state = cur_demo[i0 - 1]
                next_state = cur_demo[i0]
                action_list = self.get_action_list(cur_state)
                j = [self.get_state_transition(cur_state, a0) for a0 in action_list].index(next_state)
                action = action_list[j]
                expert_st.append(cur_state)
                expert_des.append(demo_des)
                expert_ac.append(action)
                expert_st_next.append(next_state)
        return torch.LongTensor(expert_st), torch.LongTensor(expert_des), torch.LongTensor(expert_ac), torch.LongTensor(
            expert_st_next)

    def import_demonstrations_step(self, demopath, n_rows=None):
        demo = pd.read_csv(demopath, header=0, nrows=n_rows)
        trajs = []
        for demo_str, demo_des in zip(demo['path'].tolist(), demo['des'].tolist()):
            cur_demo = [int(r) for r in demo_str.split('_')]
            len_demo = len(cur_demo)
            episode = []
            for i0 in range(1, len_demo):
                cur_state = cur_demo[i0 - 1]
                next_state = cur_demo[i0]

                action_list = self.get_action_list(cur_state)
                j = [self.get_state_transition(cur_state, a0) for a0 in action_list].index(next_state)
                action = action_list[j]

                reward = self.get_reward(cur_state)
                is_done = next_state == demo_des

                episode.append(
                    Step(cur_state=cur_state, action=action, next_state=next_state, reward=reward, done=is_done))
            trajs.append(episode)
            self.max_route_length = len(episode) if self.max_route_length < len(episode) else self.max_route_length
        print('max_route_length', self.max_route_length)
        print('n_traj', len(trajs))
        return trajs