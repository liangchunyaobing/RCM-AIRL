import editdistance
from nltk.translate.bleu_score import sentence_bleu
from scipy.spatial import distance
from nltk.translate.bleu_score import SmoothingFunction
import numpy as np
import torch

smoothie = SmoothingFunction().method1
device = torch.device("cpu")


def create_od_set(test_trajs):
    test_od_dict = {}
    for i in range(len(test_trajs)):
        if (test_trajs[i][0], test_trajs[i][-1]) in test_od_dict.keys():
            test_od_dict[(test_trajs[i][0], test_trajs[i][-1])].append(i)
        else:
            test_od_dict[(test_trajs[i][0], test_trajs[i][-1])] = [i]
    return test_od_dict


def evaluate_edit_dist(test_trajs, learner_trajs, test_od_dict):
    edit_dist_list = []
    for od in test_od_dict.keys():
        idx_list = test_od_dict[od]
        test_od_trajs = set(['_'.join(test_trajs[i]) for i in idx_list])
        test_od_trajs = [traj.split('_') for traj in test_od_trajs]
        learner_od_trajs = [learner_trajs[i] for i in idx_list]
        for learner in learner_od_trajs:
            min_edit_dist = 1.0
            for test in test_od_trajs:
                edit_dist = editdistance.eval(test, learner) / len(test)
                min_edit_dist = edit_dist if edit_dist < min_edit_dist else min_edit_dist
            edit_dist_list.append(min_edit_dist)
    return np.mean(edit_dist_list)


def evaluate_bleu_score(test_trajs, learner_trajs, test_od_dict):
    bleu_score_list = []
    for od in test_od_dict.keys():
        idx_list = test_od_dict[od]
        # get unique reference
        test_od_trajs = set(['_'.join(test_trajs[i]) for i in idx_list])
        test_od_trajs = [traj.split('_') for traj in test_od_trajs]
        learner_od_trajs = [learner_trajs[i] for i in idx_list]
        for learner in learner_od_trajs:
            # print(test_od_trajs)
            # print(learner)
            bleu_score = sentence_bleu(test_od_trajs, learner, smoothing_function=smoothie)
            bleu_score_list.append(bleu_score)
    return np.mean(bleu_score_list)


def evaluate_dataset_dist(test_trajs, learner_trajs):
    test_trajs_str = ['_'.join(traj) for traj in test_trajs]
    # print('test trajs str len', len(test_trajs_str))
    test_trajs_set = set(test_trajs_str)
    # print('test trajs set len', len(test_trajs_set))
    test_trajs_dict = dict(zip(list(test_trajs_set), range(len(test_trajs_set))))
    test_trajs_label = [test_trajs_dict[traj] for traj in test_trajs_str]
    test_trajs_label.append(0)
    test_p = np.histogram(test_trajs_label)[0] / len(test_trajs_label)

    pad_idx = len(test_trajs_set)
    learner_trajs_str = ['_'.join(traj) for traj in learner_trajs]
    learner_trajs_label = [test_trajs_dict.get(traj, pad_idx) for traj in learner_trajs_str]
    learner_p = np.histogram(learner_trajs_label)[0] / len(learner_trajs_label)
    return distance.jensenshannon(test_p, learner_p)


def evaluate_log_prob(test_traj, model):
    log_prob_list = []
    for episode in test_traj:
        des = torch.LongTensor([episode[-1].next_state]).long().to(device)
        log_prob = 0
        for x in episode:
            with torch.no_grad():
                next_prob = torch.log(model.get_action_prob(torch.LongTensor([x.cur_state]).to(device), des)).squeeze()
            next_prob_np = next_prob.detach().cpu().numpy()
            log_prob += next_prob_np[x.action]
        log_prob_list.append(log_prob)
    print(np.mean(log_prob_list))
    return np.mean(log_prob_list)


def evaluate_train_edit_dist(train_traj, learner_traj):
    """This function is used to keep the training epoch with the best edit distance performance on the training data"""
    test_od_dict = create_od_set(train_traj)
    edit_dist = evaluate_edit_dist(train_traj, learner_traj, test_od_dict)
    return edit_dist


def evaluate_metrics(test_traj, learner_traj):
    test_od_dict = create_od_set(test_traj)
    edit_dist = evaluate_edit_dist(test_traj, learner_traj, test_od_dict)
    bleu_score = evaluate_bleu_score(test_traj, learner_traj, test_od_dict)
    js_dist = evaluate_dataset_dist(test_traj, learner_traj)
    print('edit dist', edit_dist)
    print('bleu score', bleu_score)
    print('js distance', js_dist)
    return edit_dist, bleu_score, js_dist


def evaluate_model(target_od, target_traj, model, env, n_link=714):
    state_ts = torch.from_numpy(np.arange(n_link)).long().to(device)
    target_o, target_d = target_od[:, 0].tolist(), target_od[:, 1].tolist()
    learner_traj = []
    """compute transition matrix for the first OD pair"""
    curr_ori, curr_des = target_o[0], target_d[0]
    des_ts = (torch.ones_like(state_ts) * curr_des).to(device)
    action_prob = model.get_action_prob(state_ts, des_ts).detach().cpu().numpy()  # 714, 8
    state_action = env.state_action[:-1]
    action_prob[state_action == env.pad_idx] = 0.0
    transit_prob = np.zeros((n_link, n_link))
    from_st, ac = np.where(state_action != env.pad_idx)
    to_st = state_action[state_action != env.pad_idx]
    transit_prob[from_st, to_st] = action_prob[from_st, ac]
    """compute sample path for the first OD pair"""
    sample_path = [str(curr_ori)]
    curr_state = curr_ori
    for _ in range(50):
        if curr_state == curr_des: break
        next_state = np.argmax(transit_prob[curr_state])
        sample_path.append(str(next_state))
        curr_state = next_state
    learner_traj.append(sample_path)
    for ori, des in zip(target_o[1:], target_d[1:]):
        if des == curr_des:
            if ori == curr_ori:
                learner_traj.append(sample_path)
                continue
            else:
                curr_ori = ori
        else:
            curr_ori, curr_des = ori, des
            des_ts = (torch.ones_like(state_ts) * curr_des).to(device)
            action_prob = model.get_action_prob(state_ts, des_ts).detach().cpu().numpy()  # 714, 8
            state_action = env.state_action[:-1]
            action_prob[state_action == env.pad_idx] = 0.0
            transit_prob = np.zeros((n_link, n_link))
            from_st, ac = np.where(state_action != env.pad_idx)
            to_st = state_action[state_action != env.pad_idx]
            transit_prob[from_st, to_st] = action_prob[from_st, ac]
        sample_path = [str(curr_ori)]
        curr_state = curr_ori
        for _ in range(50):
            if curr_state == curr_des: break
            next_state = np.argmax(transit_prob[curr_state])
            sample_path.append(str(next_state))
            curr_state = next_state
        learner_traj.append(sample_path)
    evaluate_metrics(target_traj, learner_traj)
