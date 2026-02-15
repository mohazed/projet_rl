"""
PPO (Proximal Policy Optimization)
===================================
State-of-the-art policy gradient method that typically outperforms DQN and A2C.

Key innovations:
- Clipped surrogate objective prevents too-large policy updates
- Multiple epochs of minibatch updates on collected data
- More sample-efficient than A2C
- More stable than vanilla policy gradient methods

PPO is widely considered the best general-purpose RL algorithm for
continuous control and many other domains.
"""

import numpy as np
import gymnasium as gym

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

# ============================================================
#  CONFIG
# ============================================================

ENV_ID = "LunarLander-v3"
MODEL_NAME = "ppo_model.pt"

GAMMA = 0.99
LAMBDA = 0.95  # GAE lambda
LR = 3e-4

CLIP_EPSILON = 0.2  # PPO clipping parameter
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5

N_STEPS = 2048  # steps per update (larger than A2C)
N_EPOCHS = 10  # optimization epochs per batch
BATCH_SIZE = 64  # minibatch size

MAX_EPISODES = 2000
EVAL_EVERY = 200

# Device selection
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"[PPO] Using device: {device}")


# ============================================================
#  ACTOR-CRITIC NETWORK
# ============================================================

class ActorCritic(nn.Module):
    """
    Shared backbone with two heads:
    - Actor: outputs action probabilities
    - Critic: outputs state value V(s)
    """
    def __init__(self, obs_dim, n_actions):
        super().__init__()

        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
        )

        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        features = self.shared(x)
        policy_logits = self.actor(features)
        value = self.critic(features)
        return policy_logits, value

    def get_action_and_value(self, state, action=None):
        """Get action (or evaluate given action) and value"""
        logits, value = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, value


# ============================================================
#  GAE (Generalized Advantage Estimation)
# ============================================================

def compute_gae(rewards, values, dones, next_value, gamma, lambda_):
    """
    Compute Generalized Advantage Estimation (GAE).
    This provides better bias-variance tradeoff than simple advantages.
    """
    advantages = []
    gae = 0

    values = values + [next_value]

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lambda_ * (1 - dones[t]) * gae
        advantages.insert(0, gae)

    returns = [adv + val for adv, val in zip(advantages, values[:-1])]

    return advantages, returns


# ============================================================
#  EVALUATION
# ============================================================

def evaluate(model, env_id, device, n_episodes=5):
    """Evaluate policy without rendering"""
    env = gym.make(env_id)
    model.eval()

    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total = 0
        while not done:
            s = torch.tensor([obs], dtype=torch.float32, device=device)
            with torch.no_grad():
                logits, _ = model(s)
                probs = F.softmax(logits, dim=-1)
                action = torch.argmax(probs, dim=-1).item()
            obs, r, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total += r
        rewards.append(total)

    env.close()
    model.train()
    return np.mean(rewards), np.std(rewards)


# ============================================================
#  MAIN TRAINING LOOP
# ============================================================

def main():
    env = gym.make(ENV_ID)
    obs, info = env.reset(seed=0)

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # Create actor-critic model
    model = ActorCritic(obs_dim, n_actions).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    episode = 0
    episode_reward = 0
    state = obs

    best_eval_reward = -float('inf')

    print(f"[PPO] Starting training for {MAX_EPISODES} episodes")
    print("="*60)

    global_step = 0

    while episode < MAX_EPISODES:
        # Collect trajectories
        states = []
        actions = []
        log_probs_old = []
        rewards = []
        values = []
        dones = []

        for _ in range(N_STEPS):
            global_step += 1

            state_tensor = torch.tensor([state], dtype=torch.float32, device=device)

            with torch.no_grad():
                action, log_prob, _, value = model.get_action_and_value(state_tensor)

            action_item = action.item()
            next_state, reward, terminated, truncated, info = env.step(action_item)
            done = terminated or truncated

            states.append(state)
            actions.append(action_item)
            log_probs_old.append(log_prob.item())
            rewards.append(reward)
            values.append(value.item())
            dones.append(done)

            state = next_state
            episode_reward += reward

            if done:
                episode += 1
                print(f"Episode {episode:4d} | Reward: {episode_reward:7.1f}")

                # Periodic evaluation
                if episode % EVAL_EVERY == 0:
                    mean_reward, std_reward = evaluate(model, ENV_ID, device)
                    print(f"  → Eval: {mean_reward:.1f} ± {std_reward:.1f}")

                    if mean_reward > best_eval_reward:
                        best_eval_reward = mean_reward
                        torch.save(model.state_dict(), MODEL_NAME)
                        print(f"  → New best model saved! ({mean_reward:.1f})")

                state, info = env.reset()
                episode_reward = 0

                if episode >= MAX_EPISODES:
                    break

        if episode >= MAX_EPISODES:
            break

        # Bootstrap value for last state
        with torch.no_grad():
            next_state_tensor = torch.tensor([state], dtype=torch.float32, device=device)
            _, next_value = model(next_state_tensor)
            next_value = next_value.item()

        # Compute GAE advantages and returns
        advantages, returns = compute_gae(rewards, values, dones, next_value, GAMMA, LAMBDA)

        # Convert to tensors
        states_tensor = torch.tensor(np.array(states), dtype=torch.float32, device=device)
        actions_tensor = torch.tensor(actions, dtype=torch.int64, device=device)
        log_probs_old_tensor = torch.tensor(log_probs_old, dtype=torch.float32, device=device)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32, device=device)
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=device)

        # Normalize advantages
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

        # PPO optimization with multiple epochs
        for epoch in range(N_EPOCHS):
            # Create minibatches
            indices = np.arange(len(states))
            np.random.shuffle(indices)

            for start in range(0, len(states), BATCH_SIZE):
                end = start + BATCH_SIZE
                batch_indices = indices[start:end]

                batch_states = states_tensor[batch_indices]
                batch_actions = actions_tensor[batch_indices]
                batch_log_probs_old = log_probs_old_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]

                # Get current policy and value
                _, log_probs, entropy, values_pred = model.get_action_and_value(
                    batch_states, batch_actions
                )

                # Policy loss with clipping
                ratio = torch.exp(log_probs - batch_log_probs_old)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values_pred.squeeze(), batch_returns)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = policy_loss + VALUE_COEF * value_loss + ENTROPY_COEF * entropy_loss

                # Optimization step
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

    env.close()
    print("="*60)
    print(f"[PPO] Training complete! Best eval reward: {best_eval_reward:.1f}")
    print(f"Model saved as: {MODEL_NAME}")


if __name__ == "__main__":
    main()
