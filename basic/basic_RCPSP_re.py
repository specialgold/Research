import tensorflow as tf
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True

# config.gpu_options.per_process_gpu_memory_fraction = 0.3333
import numpy as np
from collections import deque
import basic_RCPSP_env
import NetworkBuilder as dqn
import basic_param
import RCPSP_fileparser as fp

import random

REPLAY_MEMORY = 50000

info = fp.parser('j102_2.mm')
pa = basic_param.Parameters(info)
env = basic_RCPSP_env.Env(pa)
dis = pa.discount

def replay_train(mainDQN, trajs):
    x_stack = np.empty(0).reshape(0, mainDQN.input_height, mainDQN.input_width)
    y_stack = np.empty(0).reshape(0, mainDQN.output_size)
    next_reward = 0
    for traj in trajs:
        # print('start')
        while len(traj) != 0:
            state, action, reward, next_state, done = traj.pop()
            # print(done)
            Q = mainDQN.predict(state)

            if done:
                Q[0, action] = reward
                next_reward = reward
            else:
                Q[0, action] = reward + dis * next_reward
                # Q[0, action] = reward + dis * next_reward
            # next_reward = reward
            # print('predict next_reward: ' + str(np.max(mainDQN.predict(next_state))))
            # print('next_reward'+str(next_reward))
            y_stack = np.vstack([y_stack, Q])
            x_stack = np.vstack([x_stack, state])

    # for state, action, reward, next_state, done in train_batch:
    #     Q = mainDQN.predict(state)
    #
    #     if done:
    #         Q[0, action] = reward
    #     else:
    #         # Q[0, action] = reward + dis * np.max(mainDQN.predict(next_state))
    #         Q[0, action] = reward + dis * next_reward
    #     next_reward = reward
    #
    #     y_stack = np.vstack([y_stack, Q])
    #     x_stack = np.vstack([x_stack, state])
    return mainDQN.update(x_stack, y_stack)

def main():
    max_episodes = 50000

    # replay_buffer = deque()

    # with tf.Session(config=tf.ConfigProto(device_count={'GPU':0})) as sess:
    with tf.Session() as sess:
        mainDQN = dqn.DQN(sess, pa.network_input_height, pa.network_input_width, pa.network_output_height, name="main")
        # mainDQN = dqn.DQN(sess, pa.horizon * (pa.renewable) + pa.num_queue * (pa.renewable + 1), pa.network_output_height, name="main")
        tf.compat.v1.global_variables_initializer().run()
        trajs = deque()
        trajs_rew = deque()
        for episode in range(max_episodes):
            e = 1. / ((episode / 1000) + 1)
            done = False
            step_count = 0
            state = env.observe()
            env.reset()
            info = []
            traj = []
            show_result = 0;
            while not done:
                if np.random.rand(1) < e:
                    action = env.random_action()
                    # if episode % 10 == 0:
                    #     print(info)
                    #     print(action)
                else:
                    action = np.argmax(mainDQN.predict(state))
                    # if episode % 10 == 0:
                    #     print(info)
                    #     print(mainDQN.predict(state))

                # print("action:"+str(action))
                # env.step(action)
                next_state, reward, done, info = env.step(action)
                traj.append((state, action, reward, next_state, done))

                # replay_buffer.append((state, action, reward, next_state, done))
                # if len(replay_buffer) > REPLAY_MEMORY:
                #     replay_buffer.popleft()
                if done:
                    show_result = reward
                state = next_state
                step_count += 1
                if step_count > 100:
                    break

            trajs.append(traj)
            if episode % 10 == 0:
                print("Episode: {} steps: {}  reward: {}".format(episode, step_count, show_result))
                # print(info)

            if episode % 10 == 1:
                if episode != 1:

                    # minibatch = random.sample(trajs, 10)
                    loss, _ = replay_train(mainDQN, trajs)
                    print("Loss: ", loss)
                    trajs = deque()
                    # for _ in range(50):
                    #     minibatch = random.sample(trajs, 10)
                    #     loss, _ = replay_train(mainDQN, minibatch)
                    # print("Loss: ", loss)



if __name__ == "__main__":
    main()