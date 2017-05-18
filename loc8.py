"""Loc8."""

from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import rl.core as krl
from keras.optimizers import Adam
from sklearn.cluster import MiniBatchKMeans

from model import model


class Loc8World:
    """Loc8 world simulator."""

    def __init__(self, *shape, goal=None, start=None):

        assert shape, "World must have a defined shape"

        self.explored = np.zeros((0, len(shape)), dtype=np.float)
        self.shape = np.array(shape)

        if start is None:
            start = (0,) * len(shape)
        self.position = start

        self.goal = goal if goal is not None else self.random_goal()

    @property
    def position(self):
        """Current explorer position."""
        return self.explored[-1]

    @position.setter
    def position(self, value):
        """Set explorer position."""
        self.explored = np.vstack([self.explored, value])

    @property
    def array(self):

        array = np.zeros(self.shape)

        for coord in np.floor(self.explored).astype(np.int):
            if (coord < self.shape).all():
                array[tuple(coord)] = 1

        return array

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
        return np.sqrt(((center - self.explored)**2).sum(axis=1))

    def far_points(self):

        results = []

        for coords in np.argwhere(self.array == 0):
            results.append((coords, self.distances(coords).min()))

        largest = max(results, key=lambda a: a[1])[1]
        filtered = filter(lambda a: a[1] >= largest / 2, results)

        far_points = np.array([point[0] for point in filtered])

        return far_points

    def key_points(self, n_points):
        """Use Mini-Batch K-Means clustering to develop :data:`n_clusters`
        clusters that are representative of the empty space in the
        :class:`Loc8World`."""

        far_points = self.far_points()

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

    def __init__(self, *shape, n_points, vision=3, moving_average_len=3):

        self.world = Loc8World(*shape)
        self.n_points = n_points
        self.vision = vision

        self.steps = 0
        self.observation = self.observe()

        self.choices = deque(maxlen=moving_average_len)

    def observe(self):
        """Retrieve observations for reinforcement learning."""
        return self.world.array.reshape(5, 5, 5, 5).mean(axis=(3, 1)).flatten()

    def step(self, action):

        action = np.clip(action, 0, 1) * self.world.shape
        self.choices.append(action)

        moving_towards = np.mean(self.choices, axis=0)
        delta = moving_towards - self.world.position
        if not np.all(delta == 0):
            self.world.position += delta / np.sqrt((delta**2).sum())

        observation = self.observe()

        done = self.world.distance_to_goal < self.vision

        reward = np.sum(observation - self.observation) * 10
        self.observation = observation
        self.steps += 1

        # observation, reward, done, info
        return observation, reward, done, {}

    def reset(self):

        plt.clf()
        plt.xlim(0, self.world.shape[0])
        plt.ylim(0, self.world.shape[1])

        self.world = Loc8World(*self.world.shape, goal=(5, 20))

        goal = self.world.goal
        plt.plot([goal[0]], [goal[1]], 'g^', markersize=20)

        self.steps = 0
        self.observation = self.observe()
        return self.observation

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


def run(*shape, n_points, actor_dense=1, critic_dense=1, **kwargs):
    """Run reinforcement learning algorithm with a given world size."""

    env = Loc8Env(*shape, n_points=n_points, **kwargs)
    nb_actions = len(env.world.shape)
    observation_shape = (25,)  # (np.prod(env.world.shape),)

    agent = model(
        nb_actions, observation_shape,
        actor_dense=actor_dense, critic_dense=critic_dense
    )
    agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

    agent.fit(env, nb_steps=50000, visualize=False, verbose=2)

    # agent.save_weights("loc8_weights.h5f", overwrite=True)

    agent.test(env, nb_episodes=50, visualize=True)

if __name__ == "__main__":
    run(25, 25, n_points=5, moving_average_len=10)
