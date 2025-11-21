import numpy as np
import random

class BinaryBandit:
    def __init__(self, p1=0.3, p2=0.7):
        self.probs = [p1, p2]

    def pull(self, action):
        return 1 if random.random() < self.probs[action] else 0


def epsilon_greedy_binary(bandit, epsilon=0.1, steps=10000):
    Q = [0.0, 0.0]
    N = [0, 0]
    rewards = []

    for t in range(steps):
        if random.random() < epsilon:
            action = random.randint(0, 1)
        else:
            action = 0 if Q[0] >= Q[1] else 1

        reward = bandit.pull(action)
        N[action] += 1
        Q[action] += (reward - Q[action]) / N[action]
        rewards.append(reward)

    return Q, N, rewards


class NonStationaryBandit:
    def __init__(self):
        self.means = np.zeros(10)

    def pull(self, action):
        reward = np.random.randn() + self.means[action]
        self.means += np.random.normal(0, 0.01, 10)
        return reward


def epsilon_greedy_nonstationary(bandit, epsilon=0.1, alpha=0.1, steps=10000):
    Q = np.zeros(10)
    rewards = []
    optimal_actions = []

    for t in range(steps):
        if random.random() < epsilon:
            action = random.randint(0, 9)
        else:
            action = np.argmax(Q)

        reward = bandit.pull(action)
        Q[action] += alpha * (reward - Q[action])

        rewards.append(reward)
        optimal_actions.append(action == np.argmax(bandit.means))

    return Q, rewards, optimal_actions


def test_binary_bandits():
    banditA = BinaryBandit(0.3, 0.7)
    banditB = BinaryBandit(0.6, 0.4)

    for name, bandit in [("BanditA", banditA), ("BanditB", banditB)]:
        Q, N, rewards = epsilon_greedy_binary(bandit, epsilon=0.1, steps=10000)
        print(f"\n{name}:")
        print(f"  Q-values: {Q}")
        print(f"  Counts: {N}")
        print(f"  Avg Reward: {np.mean(rewards):.3f}")
        print(f"  Best Action: {np.argmax(Q)}")


def test_nonstationary():
    bandit = NonStationaryBandit()
    Q1, r1, opt1 = epsilon_greedy_nonstationary(bandit, epsilon=0.1, alpha=1/10, steps=10000)

    bandit = NonStationaryBandit()
    Q2, r2, opt2 = epsilon_greedy_nonstationary(bandit, epsilon=0.1, alpha=0.1, steps=10000)

    print(f"Standard (1/n):  Avg Reward = {np.mean(r1):.3f}, Optimal % = {np.mean(opt1)*100:.1f}%")
    print(f"Constant (0.1):  Avg Reward = {np.mean(r2):.3f}, Optimal % = {np.mean(opt2)*100:.1f}%")


if __name__ == "__main__":
    test_binary_bandits()
    test_nonstationary()
