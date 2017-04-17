"""Loc8."""

from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import rl.core as krl
from keras.layers import LSTM, Activation, Dense, Flatten, Input, concatenate
from keras.models import Model, Sequential
from keras.optimizers import Adam
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from sklearn.cluster import MiniBatchKMeans
from sklearn.utils.extmath import cartesian


class Loc8World:
    """Loc8 world simulator."""

    def __init__(self, *shape, goal=None, start=None):

        assert shape, "World must have a defined shape"

        self.explored = np.zeros((0, len(shape)), dtype=np.float)
        self.shape = np.array(shape)

        if start is None:
            start = (0,) * len(shape)
        self.position = start

        self.goal = goal if goal is not None else (5,) * len(shape)

    @property
    def position(self):
        """Current explorer position."""
        return self.explored[-1]

    @position.setter
    def position(self, value):
        """Set explorer position."""
        self.explored = np.vstack([self.explored, value])

    @property
    def distance_to_goal(self):
        """Return the current distance to the goal."""
        return np.sqrt(((self.goal - self.position)**2).sum())

    def random_goal(self):
        """Choose a random goal."""
        self.goal = np.array([np.random.randint(axis) for axis in self.shape])
        return self.goal

    def distances(self, center):
        """Calculate distances from all explored points to a given center."""
        return np.sqrt(((center - self.explored)**2).sum(axis=1)).min()

    def key_points(self, n_points):
        """Use Mini-Batch K-Means clustering to develop :data:`n_clusters`
        clusters that are representative of the empty space in the
        :class:`Loc8World`."""

        results = []

        for coords in cartesian([np.arange(axis) for axis in self.shape]):
            results.append((coords, self.distances(coords)))

        largest = max(results, key=lambda a: a[1])[1]
        filtered = filter(lambda a: a[1] >= largest / 2, results)

        far_points = np.array([point[0] for point in filtered])

        if len(far_points) < n_points:
            far_points = np.tile(
                far_points.T,
                int(np.ceil(n_points / len(far_points)))
            ).T[:n_points]

        centers = MiniBatchKMeans(
            n_clusters=n_points
        ).fit(far_points).cluster_centers_

        return centers


class Loc8Env(krl.Env):
    """Loc8 environment."""

    reward_range = (-1, 1)

    def __init__(self, *shape, n_points, moving_average_len=3):

        self.world = Loc8World(*shape)
        self.n_points = n_points

        self.choices = deque(maxlen=moving_average_len)

    ########

    def observe(self):
        """Retrieve observations for reinforcement learning."""
        return self.world.key_points(self.n_points)

    def collect_reward(self):
        """Collect reward for reinforcement learning step."""
        return 0  # HACK
        return (1 - np.abs(self.world.surroundings(self.radius))).sum()

    ########

    def step(self, action):

        action = np.clip(action, -1, 1)

        observation = self.observe()
        print(observation)
        print(action)
        # n = np.random.choice(len(observation))  # HACK
        # self.choices.append(observation[n])  # HACK
        self.choices.append(observation[action])

        moving_towards = np.mean(self.choices, axis=0)

        # d = moving_towards - self.world.position
        # self.world.position += np.sqrt((d**2).sum())
        # self.world.position += d / 2  # np.sqrt((d**2).sum())
        self.world.position = moving_towards

        reward = self.collect_reward()

        done = self.world.distance_to_goal < 3  # TODO

        # observation, reward, done, info
        return observation, reward, done, {}

    def reset(self):

        plt.clf()
        plt.xlim(0, self.world.shape[0])
        plt.ylim(0, self.world.shape[1])

        self.world = Loc8World(*self.world.shape)

        return self.observe()

    ########

    def render(self, mode="human", close=False):

        if mode != "human":
            return

        plt.plot(*np.expand_dims(self.world.position, 0).T, 'o')
        plt.pause(0.05)

    ########

    def close(self):
        pass

    def seed(self, seed=None):
        return []

    def configure(self):
        pass

###


def run(*size, n_points):
    """Run reinforcement learning algorithm with a given world size."""

    env = Loc8Env(*size, n_points=n_points)
    nb_actions = len(env.world.shape)
    observation_shape = (n_points, len(size))  # unflattenned

    actor = Sequential()
    actor.add(Flatten(input_shape=(1,) + observation_shape))
    actor.add(Dense(16))
    actor.add(Activation('relu'))
    actor.add(Dense(16))
    actor.add(Activation('relu'))
    actor.add(Dense(16))
    actor.add(Activation('relu'))
    actor.add(Dense(nb_actions))
    actor.add(Activation('linear'))
    print(actor.summary())

    action_input = Input(shape=(nb_actions,), name='action_input')
    observation_input = Input(
        shape=(1,) + observation_shape, name='observation_input')
    flattened_observation = Flatten()(observation_input)
    x = concatenate([action_input, flattened_observation])
    x = Dense(32, activation="relu")(x)
    x = Dense(32, activation="relu")(x)
    x = Dense(32, activation="relu")(x)
    x = Dense(1, activation="tanh")(x)
    critic = Model(inputs=[action_input, observation_input], outputs=[x])
    print(critic.summary())

    memory = SequentialMemory(limit=100000, window_length=1)
    random_process = OrnsteinUhlenbeckProcess(
        size=nb_actions, theta=.15, mu=0., sigma=.3)
    agent = DDPGAgent(
        nb_actions=nb_actions,
        actor=actor,
        critic=critic,
        critic_action_input=action_input,
        memory=memory,
        nb_steps_warmup_critic=100,
        nb_steps_warmup_actor=100,
        random_process=random_process,
        gamma=.99,
        target_model_update=1e-3
    )
    agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

    agent.fit(env, nb_steps=50000, visualize=False, verbose=2)

    # agent.save_weights("loc8_weights.h5f", overwrite=True)

    agent.test(env, nb_episodes=5, visualize=True)

if __name__ == "__main__":
    run(20, 20, n_points=5)
