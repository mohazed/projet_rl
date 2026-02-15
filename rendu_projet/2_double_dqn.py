"""
DOUBLE DQN (Double Deep Q-Learning)
===================================
Addresses overestimation bias in vanilla DQN by decoupling
action selection and action evaluation:
- Policy network selects the best action
- Target network evaluates that action
This typically leads to more stable and accurate Q-value estimates.
"""

import random
import collections
import numpy as np
import gymnasium as gym

import torch
import torch.nn as nn
import torch.optim as optim

# ============================================================
#  CONFIG
# ============================================================

ENV_ID = "LunarLander-v3"
MODEL_NAME = "double_dqn_model.pt"

GAMMA = 0.99
LR = 1e-3

BUFFER_SIZE = 100_000
BATCH_SIZE = 64
MIN_REPLAY_SIZE = 10_000
TARGET_UPDATE_FREQ = 1000

MAX_EPISODES = 2000
EVAL_EVERY = 200

EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_FRAMES = 300_000

# Device selection
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"[DOUBLE DQN] Using device: {device}")


# ============================================================
#  DQN NETWORK
# ============================================================

class DQN(nn.Module):
    """Simple 3-layer MLP for Q-value approximation"""
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
#  REPLAY BUFFER
# ============================================================

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, s, a, r, ns, done):
        self.buffer.append((s, a, r, ns, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ============================================================
#  EPSILON SCHEDULE
# ============================================================

def get_epsilon(frame_idx):
    if frame_idx < MIN_REPLAY_SIZE:
        return 1.0
    ratio = min(1.0, (frame_idx - MIN_REPLAY_SIZE) / EPS_DECAY_FRAMES)
    eps = EPS_START + ratio * (EPS_END - EPS_START)
    return max(EPS_END, eps)


# ============================================================
#  EVALUATION
# ============================================================

def evaluate(policy_net, env_id, device, n_episodes=5):
    """Evaluate policy without rendering"""
    env = gym.make(env_id)
    policy_net.eval()

    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total = 0
        while not done:
            s = torch.tensor([obs], dtype=torch.float32, device=device)
            with torch.no_grad():
                q = policy_net(s)
                action = torch.argmax(q, dim=1).item()
            obs, r, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total += r
        rewards.append(total)

    env.close()
    policy_net.train()
    return np.mean(rewards), np.std(rewards)


# ============================================================
#  MAIN TRAINING LOOP
# ============================================================

def main():
    env = gym.make(ENV_ID)
    obs, info = env.reset(seed=0)

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # Create networks
    policy_net = DQN(obs_dim, n_actions).to(device)
    target_net = DQN(obs_dim, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    buffer = ReplayBuffer(BUFFER_SIZE)

    frame = 0
    episode = 0
    episode_reward = 0
    state = obs

    best_eval_reward = -float('inf')

    print(f"[DOUBLE DQN] Starting training for {MAX_EPISODES} episodes")
    print("="*60)

    while episode < MAX_EPISODES:
        frame += 1

        # Epsilon-greedy action selection
        epsilon = get_epsilon(frame)
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            s = torch.from_numpy(np.array([state], dtype=np.float32)).to(device)
            with torch.no_grad():
                q = policy_net(s)
                action = torch.argmax(q, dim=1).item()

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        buffer.push(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward

        if done:
            episode += 1
            print(f"Episode {episode:4d} | Reward: {episode_reward:7.1f} | Eps: {epsilon:.3f} | Buffer: {len(buffer)}")

            # Periodic evaluation
            if episode % EVAL_EVERY == 0:
                mean_reward, std_reward = evaluate(policy_net, ENV_ID, device)
                print(f"  → Eval: {mean_reward:.1f} ± {std_reward:.1f}")

                if mean_reward > best_eval_reward:
                    best_eval_reward = mean_reward
                    torch.save(policy_net.state_dict(), MODEL_NAME)
                    print(f"  → New best model saved! ({mean_reward:.1f})")

            state, info = env.reset()
            episode_reward = 0

        # Training step (DOUBLE DQN)
        if len(buffer) < MIN_REPLAY_SIZE:
            continue

        states, actions, rewards, next_states, dones = buffer.sample(BATCH_SIZE)

        states = torch.tensor(states, dtype=torch.float32, device=device)
        actions = torch.tensor(actions, dtype=torch.int64, device=device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=device)
        dones = torch.tensor(dones, dtype=torch.float32, device=device)

        # Current Q-values
        q_values = policy_net(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # DOUBLE DQN: Use policy_net to select action, target_net to evaluate
        with torch.no_grad():
            # Policy network selects best actions
            next_q_values_policy = policy_net(next_states)
            next_actions = torch.argmax(next_q_values_policy, dim=1)

            # Target network evaluates those actions
            next_q_values_target = target_net(next_states)
            next_q_value = next_q_values_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)

            expected_q_value = rewards + GAMMA * next_q_value * (1 - dones)

        # Loss and optimization
        loss = nn.MSELoss()(q_value, expected_q_value)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
        optimizer.step()

        # Update target network
        if frame % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(policy_net.state_dict())

    env.close()
    print("="*60)
    print(f"[DOUBLE DQN] Training complete! Best eval reward: {best_eval_reward:.1f}")
    print(f"Model saved as: {MODEL_NAME}")


if __name__ == "__main__":
    main()
