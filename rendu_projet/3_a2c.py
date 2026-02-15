"""
A2C (Advantage Actor-Critic)
=============================
Policy gradient method that uses:
- Actor: Policy network that outputs action probabilities
- Critic: Value network that estimates state values V(s)
- Advantage: A(s,a) = R - V(s) reduces variance in policy updates

Key differences from DQN:
- Learns stochastic policy directly (not Q-values)
- On-policy (uses current policy for data collection)
- No replay buffer (uses trajectories immediately)
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
MODEL_NAME = "a2c_model.pt"

GAMMA = 0.99
LR_ACTOR = 1e-4
LR_CRITIC = 1e-3

N_STEPS = 5  # number of steps before update (n-step returns)
MAX_EPISODES = 2000
EVAL_EVERY = 200

ENTROPY_COEF = 0.01  # encourages exploration

# Device selection
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"[A2C] Using device: {device}")


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

    def get_action(self, state):
        """Sample action from policy"""
        logits, value = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action.item(), log_prob, entropy, value


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
    optimizer = optim.Adam(model.parameters(), lr=LR_ACTOR)

    episode = 0
    episode_reward = 0
    state = obs

    best_eval_reward = -float('inf')

    # Storage for n-step returns
    states_batch = []
    actions_batch = []
    log_probs_batch = []
    rewards_batch = []
    values_batch = []
    entropies_batch = []
    dones_batch = []

    print(f"[A2C] Starting training for {MAX_EPISODES} episodes")
    print("="*60)

    step = 0
    while episode < MAX_EPISODES:
        step += 1

        # Get action from current policy
        state_tensor = torch.tensor([state], dtype=torch.float32, device=device)
        action, log_prob, entropy, value = model.get_action(state_tensor)

        # Step environment
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Store transition
        states_batch.append(state)
        actions_batch.append(action)
        log_probs_batch.append(log_prob)
        rewards_batch.append(reward)
        values_batch.append(value)
        entropies_batch.append(entropy)
        dones_batch.append(done)

        state = next_state
        episode_reward += reward

        # Update every N_STEPS or at episode end
        if len(states_batch) >= N_STEPS or done:
            # Compute returns and advantages
            returns = []
            advantages = []

            # Bootstrap from next state value if not terminal
            if done:
                next_value = 0
            else:
                with torch.no_grad():
                    next_state_tensor = torch.tensor([state], dtype=torch.float32, device=device)
                    _, next_value = model(next_state_tensor)
                    next_value = next_value.item()

            # Compute n-step returns
            R = next_value
            for i in reversed(range(len(rewards_batch))):
                R = rewards_batch[i] + GAMMA * R * (1 - dones_batch[i])
                returns.insert(0, R)
                advantage = R - values_batch[i].item()
                advantages.insert(0, advantage)

            # Convert to tensors
            returns = torch.tensor(returns, dtype=torch.float32, device=device)
            advantages = torch.tensor(advantages, dtype=torch.float32, device=device)
            log_probs = torch.stack(log_probs_batch)
            values = torch.cat(values_batch)
            entropies = torch.stack(entropies_batch)

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Actor loss (policy gradient with advantage)
            actor_loss = -(log_probs * advantages.detach()).mean()

            # Critic loss (MSE between predicted value and actual return)
            critic_loss = F.mse_loss(values.squeeze(), returns)

            # Entropy bonus (encourages exploration)
            entropy_loss = -entropies.mean()

            # Total loss
            loss = actor_loss + 0.5 * critic_loss + ENTROPY_COEF * entropy_loss

            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Clear batch
            states_batch = []
            actions_batch = []
            log_probs_batch = []
            rewards_batch = []
            values_batch = []
            entropies_batch = []
            dones_batch = []

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

    env.close()
    print("="*60)
    print(f"[A2C] Training complete! Best eval reward: {best_eval_reward:.1f}")
    print(f"Model saved as: {MODEL_NAME}")


if __name__ == "__main__":
    main()
