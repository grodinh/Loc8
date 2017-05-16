from keras.layers import Dense, Flatten, Input, concatenate
from keras.models import Model, Sequential
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess


def model(nb_actions, observation_shape, *,
          actor_dense=1, critic_dense=1, summary=True):

    actor = Sequential()
    actor.add(Flatten(input_shape=(1,) + observation_shape))
    for _ in range(actor_dense):
        actor.add(Dense(16, activation="relu"))
    actor.add(Dense(nb_actions, activation="linear"))
    if summary:
        actor.summary()

    action_input = Input(shape=(nb_actions,), name="action_input")
    observation_input = Input(
        shape=(1,) + observation_shape, name="observation_input")
    output = concatenate([action_input, Flatten()(observation_input)])
    for _ in range(critic_dense):
        output = Dense(16, activation="relu")(output)
    output = Dense(1, activation="linear")(output)
    critic = Model(inputs=[action_input, observation_input], outputs=[output])
    if summary:
        critic.summary()

    memory = SequentialMemory(limit=10000, window_length=1)
    random_process = OrnsteinUhlenbeckProcess(
        size=nb_actions, theta=.15, mu=0, sigma=.1)
    agent = DDPGAgent(
        nb_actions=nb_actions,
        actor=actor,
        critic=critic,
        critic_action_input=action_input,
        memory=memory,
        nb_steps_warmup_critic=50,
        nb_steps_warmup_actor=100,
        random_process=random_process,
        gamma=.99,
        target_model_update=1e-3
    )

    return agent
