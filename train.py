import torch
import torch.optim as optim
import numpy as np

def train(model, env, episodes, num_steps):
    num_inputs = env.observation_space.shape[0]
    num_outputs = env.action_space.n
    GAMMA = 0.99
    ac = model# ActorCritic(num_inputs = num_inputs, num_outputs = num_outputs, hidden = hidden)
    optimizer = optim.Adam(model.parameters(), lr = model.learning_rate)
    
    all_len = []
    avg_len = []
    all_rewards = []
    losses = []
    entropy = 0

    for episode in range(episodes):
        log_probs = []
        values = []
        rewards = []
        
        state = env.reset()
        for steps in range(num_steps):
            value, p_dist = ac(state)
            value = value.detach().numpy()
            dist = p_dist.detach().numpy()

            action = np.random.choice(num_outputs, p = np.squeeze(dist))
            log_prob = torch.log(p_dist.squeeze(0)[action])
            ent = -np.sum(np.mean(dist) * np.log(dist))
            
            state_, reward, done, _ = env.step(action)

            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)
            entropy += ent
            state = state_

            if done or steps == num_steps - 1:
                Q, _ = ac(state_)
                Q = Q.detach().numpy()
                all_rewards.append(np.sum(rewards))
                all_len.append(steps)
                avg_len.append(np.mean(all_len[-10:]))
                
                if episode % 10 == 0:
                    print('episode: {}, reward: {}, total length: {}, average length: {}'.format(episode, np.sum(rewards), steps, avg_len[-1]))
                break

        Qvals = np.zeros_like(values)
        
        for i, j in enumerate(reversed(rewards)):
            Q = j + GAMMA + Q
            Qvals[i] = Q

        values = torch.Tensor(values)
        Qvals = torch.Tensor(Qvals)
        log_probs = torch.stack(log_probs)
        #import pdb; pdb.set_trace()
        advantage = Qvals - values
        actor_loss = torch.matmul(-log_probs, advantage).mean()
        critic_loss = 0.5 * advantage.pow(2).mean()
        ac_loss = actor_loss + critic_loss + 0.001 * entropy

        losses.append(ac_loss)

        optimizer.zero_grad()
        ac_loss.backward()
        optimizer.step()

    return rewards, losses, avg_len, all_len
