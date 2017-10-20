from keras.layers import Conv3D, Dense, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import BoltzmannQPolicy


def model(nb_actions, observation_shape, *, dense_layers=16, summary=True):

    model = Sequential()
    model.add(Conv3D(
        64,
        kernel_size=(1, 10, 10),
        activation='relu',
        input_shape=(1,) + observation_shape
    ))
    model.add(Flatten())
    for _ in range(dense_layers):
        model.add(Dense(256, activation='relu'))
    model.add(Dense(nb_actions, activation='softmax'))

    model.summary()

    agent = DQNAgent(
        model=model,
        nb_actions=nb_actions,
        memory=SequentialMemory(limit=10000, window_length=1),
        nb_steps_warmup=100,
        policy=BoltzmannQPolicy()
    )
    agent.compile(Adam(lr=1e-3), metrics=['mse'])

    return agent
