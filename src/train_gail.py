import numpy as np
from model.policy import PolicyCNN
from model.value import ValueCNN
from model.discriminator import DiscriminatorCNN
import torch
from torch import nn
import math
import time
from network_env import RoadWorld
from core.ppo import ppo_step
from core.common import estimate_advantages
from core.agent import Agent
from utils.torch import to_device
from utils.evaluation import evaluate_model, evaluate_log_prob, evaluate_train_edit_dist
from utils.load_data import ini_od_dist, load_path_feature, load_link_feature, \
    minmax_normalization, load_train_sample, load_test_traj


def update_params(batch, i_iter):
    states = torch.from_numpy(np.stack(batch.state)).long().to(device)
    next_states = torch.from_numpy(np.stack(batch.next_state)).long().to(device)
    destinations = torch.from_numpy(np.stack(batch.destination)).long().to(device)
    actions = torch.from_numpy(np.stack(batch.action)).long().to(device)
    bad_masks = torch.from_numpy(np.stack(batch.bad_mask)).long().to(device)
    masks = torch.from_numpy(np.stack(batch.mask)).long().to(device)
    rewards = discrim_net.calculate_reward(states, destinations, actions).squeeze()
    with torch.no_grad():
        # values = policy_net.get_value(states, destinations)
        values = value_net(states, destinations)
        next_values = value_net(next_states, destinations)
        fixed_log_probs = policy_net.get_log_prob(states, destinations, actions)

    """get advantage estimation from the trajectories"""
    advantages, returns = estimate_advantages(rewards, masks, bad_masks, values, next_values, gamma, tau, device)

    """update discriminator"""
    e_o, g_o, discrim_loss = None, None, None
    for _ in range(epoch_disc):
        # randomly select a batch from expert_traj
        indices = torch.from_numpy(np.random.choice(expert_st.shape[0], min(states.shape[0], expert_st.shape[0]),
                                                    replace=False)).long()
        # print('indices', indices.shape)
        s_expert_st = expert_st[indices].to(device)
        s_expert_des = expert_des[indices].to(device)
        s_expert_ac = expert_ac[indices].to(device)
        # expert_state_actions = torch.from_numpy(expert_traj).long().to(device)
        g_o = discrim_net(states, destinations, actions)
        e_o = discrim_net(s_expert_st, s_expert_des, s_expert_ac)
        optimizer_discrim.zero_grad()
        discrim_loss = discrim_criterion(g_o, torch.ones((states.shape[0], 1), device=device)) + \
                       discrim_criterion(e_o, torch.zeros((s_expert_st.shape[0], 1), device=device))
        discrim_loss.backward()
        optimizer_discrim.step()

    #     expert_acc = ((e_o < 0.5).float()).mean()
    #     learner_acc = ((g_o > 0.5).float()).mean()
    # print(i_iter, 'expert_acc', expert_acc.item(), 'learner_acc', learner_acc.item(), 'loss', discrim_loss.item())
    # print('done update discriminator...')

    """perform mini-batch PPO update"""
    value_loss, policy_loss = 0, 0
    optim_iter_num = int(math.ceil(states.shape[0] / optim_batch_size))
    for _ in range(optim_epochs):
        value_loss, policy_loss = 0, 0
        perm = np.arange(states.shape[0])
        np.random.shuffle(perm)
        perm = torch.LongTensor(perm).to(device)

        states, destinations, actions, returns, advantages, fixed_log_probs = \
            states[perm].clone(), destinations[perm].clone(), actions[perm].clone(), returns[perm].clone(), advantages[
                perm].clone(), \
            fixed_log_probs[perm].clone()

        for i in range(optim_iter_num):
            ind = slice(i * optim_batch_size, min((i + 1) * optim_batch_size, states.shape[0]))
            states_b, destinations_b, actions_b, advantages_b, returns_b, fixed_log_probs_b = \
                states[ind], destinations[ind], actions[ind], advantages[ind], returns[ind], fixed_log_probs[ind]
            batch_value_loss, batch_policy_loss = ppo_step(policy_net, value_net, optimizer_policy, optimizer_value, 10,
                                                           states_b, destinations_b, actions_b, returns_b,
                                                           advantages_b, fixed_log_probs_b, clip_epsilon, l2_reg,
                                                           max_grad_norm)

            value_loss += batch_value_loss.item()
            policy_loss += batch_policy_loss.item()
    return discrim_loss.item(), value_loss, policy_loss
    # print('done update policy and value...')


def save_model(model_path):
    policy_statedict = policy_net.state_dict()
    value_statedict = value_net.state_dict()
    discrim_statedict = discrim_net.state_dict()

    outdict = {"Policy": policy_statedict,
               "Value": value_statedict,
               "Discrim": discrim_statedict}
    torch.save(outdict, model_path)


def load_model(model_path):
    model_dict = torch.load(model_path)
    policy_net.load_state_dict(model_dict['Policy'])
    print("Policy Model loaded Successfully")
    value_net.load_state_dict(model_dict['Value'])
    print("Policy Model loaded Successfully")
    discrim_net.load_state_dict(model_dict['Discrim'])
    print("Discrim Model loaded Successfully")


def main_loop():
    best_edit = 1.0
    for i_iter in range(1, max_iter_num + 1):
        """generate multiple trajectories that reach the minimum batch_size"""
        discrim_net.to(torch.device('cpu'))
        batch, log = agent.collect_samples(min_batch_size, mean_action=False)
        discrim_net.to(device)
        update_params(batch, i_iter)

        if i_iter % log_interval == 0:
            learner_trajs = agent.collect_routes_with_OD(test_od, mean_action=True)
            edit_dist = evaluate_train_edit_dist(test_trajs, learner_trajs)
            if edit_dist < best_edit:
                best_edit = edit_dist
                save_model(model_p)


def hard_update(target, source):
    """
    Copies the parameters from source network to target network
    :param target: Target network (PyTorch)
    :param source: Source network (PyTorch)
    :return:
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


if __name__ == '__main__':
    log_std = -0.0  # log std for the policy (default: -0.0)
    gamma = 0.99  # discount factor (default: 0.99)
    tau = 0.95  # gae (default: 0.95) GAE: generalized
    l2_reg = 1e-3  # l2 regularization regression (default: 1e-3)
    learning_rate = 3e-4  # gae (default: 3e-4)
    clip_epsilon = 0.2  # clipping epsilon for PPO
    num_threads = 4  # number of threads for agent (default: 4)
    min_batch_size = 8192  # 8192  # minimal batch size per PPO update (default: 2048)
    eval_batch_size = 8192  # 8192  # minimal batch size for evaluation (default: 2048)
    log_interval = 10  # interval between training status logs (default: 10)
    save_mode_interval = 50  # interval between saving model (default: 0, means don't save)
    max_grad_norm = 10 # max grad norm for ppo updates
    seed = 1 # random seed for parameter initialization
    epoch_disc = 1  # optimization epoch number for discriminator
    optim_epochs = 10  # optimization epoch number for PPO
    optim_batch_size = 64  # optimization batch size for PPO
    cv = 0  # cross validation process [0, 1, 2, 3, 4]
    size = 100  # size of training data [100, 1000, 10000]
    max_iter_num = 1000  # maximal number of main iterations {100size: 1000, 1000size: 2000, 10000size: 3000}
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    """environment"""
    edge_p = "../data/edge.txt"
    network_p = "../data/transit.npy"
    path_feature_p = "../data/feature_od.npy"
    train_p = "../data/cross_validation/train_CV%d_size%d.csv" % (cv, size)
    test_p = "../data/cross_validation/test_CV%d.csv" % cv
    model_p = "../trained_models/gail_CV%d_size%d.pt" % (cv, size)
    """inialize road environment"""
    od_list, od_dist = ini_od_dist(train_p)
    env = RoadWorld(network_p, edge_p, pre_reset=(od_list, od_dist))
    """load path-level and link-level feature"""
    path_feature, path_max, path_min = load_path_feature(path_feature_p)
    edge_feature, link_max, link_min = load_link_feature(edge_p)
    path_feature = minmax_normalization(path_feature, path_max, path_min)
    path_feature_pad = np.zeros((env.n_states, env.n_states, path_feature.shape[2]))
    path_feature_pad[:path_feature.shape[0], :path_feature.shape[1], :] = path_feature
    edge_feature = minmax_normalization(edge_feature, link_max, link_min)
    edge_feature_pad = np.zeros((env.n_states, edge_feature.shape[1]))
    edge_feature_pad[:edge_feature.shape[0], :] = edge_feature
    """seeding"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    """define actor and critic"""
    policy_net = PolicyCNN(env.n_actions, env.policy_mask,
                           env.state_action, path_feature_pad, edge_feature_pad,
                           path_feature_pad.shape[-1] + edge_feature_pad.shape[-1] + 1,
                           env.pad_idx).to(device)
    value_net = ValueCNN(path_feature_pad, edge_feature_pad,
                         path_feature_pad.shape[-1] + edge_feature_pad.shape[-1]).to(device)
    discrim_net = DiscriminatorCNN(env.n_actions, env.policy_mask,
                                   env.state_action, path_feature_pad, edge_feature_pad,
                                   path_feature_pad.shape[-1] + edge_feature_pad.shape[-1] + 1,
                                   env.pad_idx).to(device)
    policy_net.to_device(device)
    value_net.to_device(device)
    discrim_net.to_device(device)
    discrim_criterion = nn.BCELoss()
    optimizer_policy = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)
    optimizer_value = torch.optim.Adam(value_net.parameters(), lr=learning_rate)
    optimizer_discrim = torch.optim.Adam(discrim_net.parameters(), lr=learning_rate)
    """load expert trajectory"""
    expert_st, expert_des, expert_ac, expert_next_st = env.import_demonstrations(train_p)
    to_device(device, expert_st, expert_des, expert_ac, expert_next_st)
    """load expert trajectory"""
    test_trajs, test_od = load_train_sample(train_p)
    """create agent"""
    agent = Agent(env, policy_net, device, custom_reward=None, num_threads=num_threads)
    # """Train model"""
    # start_time = time.time()
    # main_loop()
    # print('train time', time.time() - start_time)
    """Evaluate model"""
    load_model(model_p)
    test_trajs, test_od = load_test_traj(test_p)
    start_time = time.time()
    evaluate_model(test_od, test_trajs, policy_net, env)
    print('test time', time.time() - start_time)
    """Evaluate log prob"""
    test_trajs = env.import_demonstrations_step(test_p)
    evaluate_log_prob(test_trajs, policy_net)