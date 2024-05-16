import gym
import numpy as np
import matplotlib.pyplot as plt

class MonteCarloLearning:
    def __init__(self, env, gamma, alpha, epsilon, numberEpisodes, numberOfBins, lowerBounds, upperBounds):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.numberEpisodes = numberEpisodes
        self.numberOfBins = numberOfBins
        self.lowerBounds = lowerBounds
        self.upperBounds = upperBounds
        self.actionNumber = env.action_space.n
        self.sumRewardsEpisode = []
        self.Qmatrix = np.random.uniform(low=0, high=1, size=(
            numberOfBins[0], numberOfBins[1], numberOfBins[2], numberOfBins[3], self.actionNumber))
        self.Returns = {}

    def returnIndexState(self, state):
        position = state[0]
        velocity = state[1]
        angle = state[2]
        angularVelocity = state[3]

        cartPositionBin = np.linspace(self.lowerBounds[0], self.upperBounds[0], self.numberOfBins[0])
        cartVelocityBin = np.linspace(self.lowerBounds[1], self.upperBounds[1], self.numberOfBins[1])
        poleAngleBin = np.linspace(self.lowerBounds[2], self.upperBounds[2], self.numberOfBins[2])
        poleAngleVelocityBin = np.linspace(self.lowerBounds[3], self.upperBounds[3], self.numberOfBins[3])

        indexPosition = np.maximum(np.digitize(position, cartPositionBin) - 1, 0)
        indexVelocity = np.maximum(np.digitize(velocity, cartVelocityBin) - 1, 0)
        indexAngle = np.maximum(np.digitize(angle, poleAngleBin) - 1, 0)
        indexAngularVelocity = np.maximum(np.digitize(angularVelocity, poleAngleVelocityBin) - 1, 0)

        return tuple([indexPosition, indexVelocity, indexAngle, indexAngularVelocity])

    def selectAction(self, state_index, index):
        if np.random.random() < self.epsilon or index < 100:
            return np.random.choice(self.actionNumber)
        return np.argmax(self.Qmatrix[state_index])

    # def simulateEpisodes(self):
    #     for indexEpisode in range(self.numberEpisodes):
    #         episode_data = []
    #         (state, _) = self.env.reset()
    #         state_index = self.returnIndexState(state)
    #         terminalState = False
    #         while not terminalState:
    #             action = self.selectAction(state_index, indexEpisode)
    #             next_state, reward, terminalState, _, _ = self.env.step(action)
    #             episode_data.append((state_index, action, reward))
    #             state_index = self.returnIndexState(next_state)
    #
    #         G = 0
    #         for state_index, action, reward in reversed(episode_data):
    #             G = reward + self.gamma * G
    #             sa_pair = (state_index, action)
    #             if sa_pair not in self.Returns:
    #                 self.Returns[sa_pair] = []
    #             self.Returns[sa_pair].append(G)
    #             self.Qmatrix[state_index + (action,)] = np.mean(self.Returns[sa_pair])
    #         self.sumRewardsEpisode.append(sum(x[2] for x in episode_data))
    #         if (indexEpisode + 1) % 100 == 0:
    #             print(f"Episode {indexEpisode + 1}: Total reward = {self.sumRewardsEpisode[-1]}")

    def simulateEpisodes(self):
        for indexEpisode in range(self.numberEpisodes):
            episode_data = []
            (state, _) = self.env.reset()
            state_index = self.returnIndexState(state)
            terminalState = False

            n = 0
            while not terminalState:
                action = self.selectAction(state_index, indexEpisode)
                next_state, reward, terminalState, _, _ = self.env.step(action)
                episode_data.append((state_index, action, reward))
                state_index = self.returnIndexState(next_state)
                n += 1

            G = 0
            for state_index, action, reward in reversed(episode_data):
                G += reward * (self.gamma ** n)
                self.Qmatrix[state_index + (action,)] += self.alpha * (G - self.Qmatrix[state_index + (action,)])
                n -= 1

            self.sumRewardsEpisode.append(sum(x[2] for x in episode_data))
            if (indexEpisode + 1) % 100 == 0:
                print(f"Episode {indexEpisode + 1}: Total reward = {self.sumRewardsEpisode[-1]}")

    def simulateLearnedStrategy(self):
        env1 = gym.make('CartPole-v1', render_mode='human')
        currentState, _ = env1.reset()
        timeSteps = 1000
        obtainedRewards = []

        for timeIndex in range(timeSteps):
            currentStateIndex = self.returnIndexState(currentState)
            actionInStateS = np.argmax(self.Qmatrix[currentStateIndex])
            currentState, reward, terminated, _, _ = env1.step(actionInStateS)
            obtainedRewards.append(reward)
            if terminated:
                break
        env1.close()
        return obtainedRewards

# Environment setup
env = gym.make('CartPole-v1')

alpha = 0.1
gamma = 0.95
epsilon = 0.2
number_episodes = 500

lower_bounds = env.observation_space.low
upper_bounds = env.observation_space.high
upper_bounds[1] = 3
upper_bounds[3] = 2
lower_bounds[1] = -3
lower_bounds[3] = -2

number_bins = [10, 10, 10, 10]

# Create Monte Carlo instance
MC = MonteCarloLearning(env, gamma, alpha, epsilon, number_episodes, number_bins, lower_bounds, upper_bounds)
MC.simulateEpisodes()

# Plotting
plt.figure(figsize=(12, 5))
plt.plot(MC.sumRewardsEpisode, color='blue', linewidth=1)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.yscale('log')
plt.show()
plt.savefig('convergence.png')

# Simulate learned strategy
obtainedRewards = MC.simulateLearnedStrategy()
