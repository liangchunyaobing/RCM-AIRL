import torch
from src.utils.torch import to_device


def estimate_advantages(rewards, masks, bad_masks, values, next_values, gamma, tau, device):
    rewards, masks, bad_masks, values, next_values = to_device(torch.device('cpu'),
                                                               rewards, masks, bad_masks,
                                                               values, next_values)
    tensor_type = type(rewards)
    deltas = tensor_type(rewards.size(0), 1)
    advantages = tensor_type(rewards.size(0), 1)

    # prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        deltas[i] = rewards[i] + gamma * next_values[i] - values[i] # 这一步应该是需要的 因为我等于是到这一步之后 还有一个action 需要这个action是对的才行
        advantages[i] = deltas[i] + gamma * tau * prev_advantage * masks[i]

        advantages[i] = advantages[i] * bad_masks[i]

        # prev_value = values[i, 0]
        prev_advantage = advantages[i, 0]

    returns = values + advantages
    advantages = (advantages - advantages.mean()) / advantages.std()

    advantages, returns = to_device(device, advantages, returns)
    return advantages, returns