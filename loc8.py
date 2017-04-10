"""Loc8."""

import argparse

import numpy as np
from keras.layers import LSTM, Activation, Dense
from keras.models import Sequential
from keras.optimizers import Adam

import rl.core as krl
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy


class Loc8World:
    """Loc8 world simulator."""

    def __init__(self, *size, explored=None, start=None):

        assert size or explored is not None

        if explored is None:
            self.explored = np.zeros(size, dtype=np.int)
            self.size = np.array(size)
        else:
            self.explored = explored
            self.size = np.array(self.explored.shape)

        if start is None:
            self.position = np.zeros([len(self.size)], dtype=np.int)
        else:
            self.position = np.array(start)

        self.move(*(0,) * len(self.size))

    def surroundings(self, radius):
        """Retrieve surroundings within a given radius."""

        radius = radius if radius is not None else radius
        return np.pad(
            self.explored, radius, "constant", constant_values=-1
        )[tuple(slice(n, n + 1 + 2 * radius) for n in self.position)]

    def move(self, *delta):
        """Move turtle."""

        assert len(delta) == self.position.shape[0], "Invalid delta shape"
        assert all(d in (-1, 0, 1) for d in delta), "Too much movement"

        position = self.position.copy()

        position += delta
        position = np.array([
            max(0, min(p, s - 1)) for p, s in zip(position, self.size)
        ])

        if self.explored[tuple(position)] == -1:
            return self.position

        self.position = position
        self.explored[tuple(self.position)] = True
        return self.position


class Loc8Env(krl.Env):
    """Loc8 environment."""

    reward_range = (-1, 1)

    def __init__(self, *size, radius=1):

        assert size

        self.world = Loc8World(*size)
        self.radius = radius

        dims = len(self.world.size)

        self.directions = np.array(
            np.meshgrid(*[[-1, 0, 1]] * dims)
        ).T.reshape(-1, dims)

    ########

    def observe(self):
        """Retrieve observations for reinforcement learning."""
        return self.world.surroundings(self.radius).flatten()

    def collect_reward(self):
        """Collect reward for reinforcement learning step."""
        return (1 - np.abs(self.world.surroundings(self.radius))).sum()

    ########

    def step(self, action):

        self.world.move(*self.directions[action])

        observation = self.observe()
        reward = self.collect_reward()
        done = (observation == 100).any()  # HACK

        if done:
            reward = 100

        # observation, reward, done, info
        return observation, reward, done, {}

    def reset(self):

        size = self.world.size

        explored = np.zeros(size, dtype=np.int)

        goal = (0, 0)
        while goal == (0, 0):
            goal = tuple(np.random.randint(n) for n in explored.shape)
        explored[goal] = 100

        self.world = Loc8World(*self.world.size, explored=explored)
        return self.observe()

    ########

    def render(self, mode="human", close=False):

        if mode != "human":
            return

        if self.world.explored.ndim == 1:
            self.render_1d()
        elif self.world.explored.ndim == 2:
            self.render_2d()

    def render_1d(self):
        """Render a one-dimensional world using ASCII."""
        for i, col in enumerate(self.world.explored):
            if i == self.world.position:
                print("●", end='')
            elif self.world.explored[i] == 100:
                print("X", end='')
            else:
                print(("□", " ", "█")[col + 1], end='')
        print('|')

    def render_2d(self):
        """Render a two-dimensional world using ASCII."""
        for i, row in enumerate(self.world.explored):
            for j, col in enumerate(row):
                if ([i, j] == self.world.position).all():
                    print("●", end='')
                elif self.world.explored[i, j] == 100:
                    print("X", end='')
                else:
                    print(("□", " ", "█")[col + 1], end='')
            print('|')

    ########

    def close(self):
        print("close")

    def seed(self, seed=None):
        return []

    def configure(self):
        print("configure")

###


def run(*size, radius, nb_dense=8, dense_output=16):
    """Run reinforcement learning algorithm with a given world size."""

    env = Loc8Env(*size, radius=radius)
    nb_actions = len(env.directions)
    observation_shape = ((2 * radius + 1)**2,)  # flattenned

    model = Sequential()
    model.add(LSTM(2, input_shape=(1,) + observation_shape))

    for _ in range(nb_dense):
        model.add(Dense(dense_output))

    model.add(Dense(nb_actions))  # Desired output shape
    model.add(Activation("linear"))
    print(model.summary())

    memory = SequentialMemory(limit=50000, window_length=1)
    policy = EpsGreedyQPolicy(eps=.1)
    dqn = DQNAgent(
        model=model,
        nb_actions=nb_actions,
        memory=memory,
        nb_steps_warmup=10,
        target_model_update=1e-2,
        policy=policy
    )
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    dqn.fit(env, nb_steps=50000, visualize=False, verbose=2)

    # dqn.save_weights("loc8_weights.h5f", overwrite=True)

    dqn.test(env, nb_episodes=5, visualize=True)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Loc8 Engine")
    parser.add_argument(
        "--nb_dense", help="the number of dense layers to use",
        type=int, default=8
    )
    parser.add_argument(
        "--dense_output", help="the number of nodes per dense layer",
        type=int, default=16
    )

    args = parser.parse_args()

    run(20, 20, radius=2, **args.__dict__)
