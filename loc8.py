"""Loc8."""

import matplotlib.pyplot as plt
import numpy as np
import rl.core as krl

from model import model

LAYERS = 3


class Loc8World:
    """Loc8 world simulator."""

    def __init__(self, *shape, goal=None, start=None):

        assert shape, "World must have a defined shape"

        self.shape = np.array(shape)

        self.position = np.array(start) if start else np.zeros(len(shape))
        self.array = np.zeros((*self.shape, LAYERS))

        self._goal = goal
        self.goal = self._goal or self.random_goal()

    def move(self, delta):
        copy = self.position.copy()
        self.position += delta
        self.position = self.position.clip(0, self.shape - 1)
        return copy

    def coords(self, position=None):
        if position is None:
            position = self.position
        return tuple(position.astype(np.int))

    @property
    def distance_to_goal(self):
        """Return the current distance to the goal."""
        return np.linalg.norm(self.goal - self.position)

    def random_goal(self):
        """Choose a random goal."""
        return np.array([np.random.randint(axis) for axis in self.shape])


class Loc8Env(krl.Env):
    """Loc8 environment."""

    def __init__(self, *shape, vision=2, goal=None):

        self.world = Loc8World(*shape, goal=goal)
        self.vision = vision

        self.choices = np.array(
            np.meshgrid(*[[-1, 0, 1]] * len(shape))
        ).T.reshape(-1, len(shape))

    def observe(self):
        """Retrieve observations for reinforcement learning."""
        coords = self.world.coords()

        self.world.array[(*coords, 1)] = 1
        world = self.world.array.copy()

        world[(*coords, 2)] = 1

        return world

    def step(self, action):

        reward = 0

        choice = self.choices[action]
        self.world.move(choice)

        observation = self.observe()

        done = self.world.distance_to_goal < self.vision

        reward = 100 if done else 0

        # observation, reward, done, info
        return observation, reward, done, {}

    def reset(self):

        plt.clf()
        plt.xlim(0, self.world.shape[0])
        plt.ylim(0, self.world.shape[1])

        self.world = Loc8World(*self.world.shape, goal=self.world._goal)

        goal = self.world.goal
        plt.plot([goal[0]], [goal[1]], 'g^', markersize=20)

        return self.observe()

    ########

    def render(self, mode="human", close=False):

        if mode != "human":
            return

        plt.plot(*np.expand_dims(self.world.position, 0).T, 'o')
        plt.pause(0.01)

    ########

    def close(self):
        pass

###


def run(*shape, dense_layers=16, **kwargs):
    """Run reinforcement learning algorithm with a given world size."""

    env = Loc8Env(*shape, **kwargs)
    nb_actions = len(env.choices)
    observation_shape = (*env.world.shape, LAYERS)

    agent = model(
        nb_actions, observation_shape,
        dense_layers=dense_layers
    )

    agent.fit(
        env,
        nb_steps=50000,
        nb_max_episode_steps=100,
        visualize=True,
        verbose=1,
    )

    agent.test(
        env,
        nb_episodes=50,
        nb_max_episode_steps=100,
        visualize=True,
    )

if __name__ == "__main__":
    run(10, 10, goal=(2, 8))
