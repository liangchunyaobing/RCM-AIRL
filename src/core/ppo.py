import torch


def ppo_step(policy_net, value_net, optimizer_policy, optimizer_value, optim_value_iternum, states, destinations,
             actions, returns, advantages, fixed_log_probs, clip_epsilon, l2_reg, max_grad_norm):
    value_loss, policy_surr = None, None
    """update critic"""
    for _ in range(optim_value_iternum):
        values_pred = value_net(states, destinations)
        value_loss = (values_pred - returns).pow(2).mean()
        optimizer_value.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_grad_norm)
        optimizer_value.step()

    """update policy"""
    log_probs = policy_net.get_log_prob(states, destinations, actions)
    ratio = torch.exp(log_probs - fixed_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    policy_surr = -torch.min(surr1, surr2).mean()
    optimizer_policy.zero_grad()
    policy_surr.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_grad_norm)
    optimizer_policy.step()

    return value_loss, policy_surr