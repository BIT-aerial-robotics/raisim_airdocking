import time
from abc import ABC
import os
from AquaML.rlalgo.CompletePolicy import CompletePolicy
from AquaML.rlalgo.TestPolicy import TestPolicy
import tensorflow as tf

import gym
from AquaML.DataType import DataInfo
from AquaML.core.RLToolKit import RLBaseEnv
import numpy as np
from env_air_sb3.AMP_sample import AMP
from env_air_sb3.env_params import rew_coeff_sou

# LD_LIBRARY_PATH= $LD_LIBRARY_PATH:/home/ming/raisim/raisim_workspace/raisimLib/raisim/linux/lib;
# PYTHONPATH=$PYTHONPATH:/home/ming/raisim/raisim_workspace/raisimLib/raisim/linux/lib


class Actor_net(tf.keras.Model):

    def __init__(self):
        super(Actor_net, self).__init__()

        self.dense1 = tf.keras.layers.Dense(512, activation='relu')
        self.dense2 = tf.keras.layers.Dense(256, activation='relu')
        self.action_layer = tf.keras.layers.Dense(8)
        # self.log_std = tf.keras.layers.Dense(1)

        # self.learning_rate = 2e-5

        self.output_info = {'action': (8,), }

        self.input_name = ('obs',)

        self.optimizer_info = {
            'type': 'Adam',
            'args': {'learning_rate': 3e-4,
                     # 'epsilon': 1e-54
                     # 'clipnorm': 0.5,
                     },
        }

    @tf.function
    def call(self, obs, mask=None):
        x = self.dense1(obs)
        x = self.dense2(x)
        action = self.action_layer(x)
        # log_std = self.log_std(x)

        return (action,)

    def reset(self):
        pass

class AirD(RLBaseEnv):
    def __init__(self, env_name="AirDocking", destandardize_info_path=None):
        super().__init__()
        # TODO: update in the future
        self.step_s = 0
        self.env = AMP(rew_coeff=rew_coeff_sou,
                       # sense_noise='self_define',
                       # sense_noise='default',
                       max_step=2000,
                       control_name="forceThrustOmega"
                       )

        self.env_name = env_name

        # our frame work support POMDP env
        self._obs_info = DataInfo(
            names=('obs', 'step',),
            shapes=((42,), (1,)),
            dtypes=np.float32
        )

        self._reward_info = ['total_reward', 'indicate_1']

        if destandardize_info_path is not None:
            self.destandardize = True
            self.destandardize_info = {'mean_obs': [],
                                       'std_obs': [],
                                       'mean_action': [],
                                       'std_action': [],
                                        }
            for key in self.destandardize_info:
                path = os.path.join(destandardize_info_path, f'{key}'+'.npy')
                self.destandardize_info[key] = np.load(path)
        else:
            self.destandardize = False


    def reset(self):

        observation = self.env.reset()
        observation = observation.reshape(1, -1)

        if self.destandardize:
            observation = (observation - self.destandardize_info['mean_obs']) / (self.destandardize_info['std_obs'] + 1e-8)

        self.step_s = 0

        # observation = observation.

        # observation = tf.convert_to_tensor(observation, dtype=tf.float32)

        obs = {'obs': observation, 'step': self.step_s}

        obs = self.initial_obs(obs)

        time.sleep(1)

        return obs, True  # 2.0.1 new version

    def step(self, action_dict):
        #time.sleep(0.03)
        self.step_s += 1
        action = action_dict['action']
        if isinstance(action, tf.Tensor):
            action = action.numpy()
        # action *= 2

        if self.destandardize:
            action = action * (self.destandardize_info['std_action'] + 1e-8) + self.destandardize_info['mean_action']

        observation, reward, done, info = self.env.step(action[0])
        observation = observation.reshape(1, -1)

        if self.destandardize:
            observation = (observation - self.destandardize_info['mean_obs']) / (self.destandardize_info['std_obs'] + 1e-8)


        indicate_1 = reward
        #
        if reward <= -100:
            reward = -1
            done = True

        obs = {'obs': observation, 'step': self.step_s}

        obs = self.check_obs(obs, action_dict)

        reward = {'total_reward': reward, 'indicate_1': indicate_1}

        # if self.id == 0:
        #     print('reward', reward)

        return obs, reward, done, info

    def close(self):
        self.env.close()


env = AirD(
            # destandardize_obs_info_path='/home/ming/aaa/AquaML-2.2.0/dataset/Joint200/normalize_info.npy',
            # destandardize_info_path='/media/ming/新加卷/aaa/models/TD3BC/TD3BCParam/2k',
            )
osb_shape_dict = env.obs_info.shape_dict

policy = CompletePolicy(
    actor=Actor_net,
    obs_shape_dict=osb_shape_dict,
    # checkpoint_path='/home/ming/aaa/AquaML-2.2.0/Tutorial2/debug_td3bcaird_suc/history_model/TD3BC/15000',
    # checkpoint_path='/media/ming/新加卷/aaa/models/TD3BC/JOINT5200/149000',
    checkpoint_path='/home/ming/aaa/AquaML-2.2.0/Tutorial2/debug_iqlaird_joint/history_model/IQL/83000',
    # checkpoint_path='/media/ming/新加卷/aaa/models/ppo/1191',
    # checkpoint_path='/home/ming/aaa/AquaML-2.2.0/Tutorial2/debug_iqlaird_test1/history_model/IQL/900',
    using_obs_scale=False,
)

test_policy = TestPolicy(
    env=env,
    policy=policy,
)

test_policy.collect(
    episode_num=100,
    episode_steps=2000,
    data_path='/home/ming/aaa/AquaML-2.2.0/Tutorial2/debug_iqlaird_suc1/test_info'
)
#  print(np.mean(test_policy.step_info))