import sys
from tqdm import tqdm
import numpy as np
from AMP_sample import AMP
from env_params import rew_coeff_sou
import time
import os

class Dataset:
    def __init__(self, cache=False):
        self.buffer = {'obs': [],
                       'action': [],
                       'total_reward': [],
                       'next_obs': [],
                       'mask': [],}
        self.cache = cache

    def init(self):
        pass

    def add(self, data: dict):
        if self.cache:
            for key in data:
                self.buffer[key].append(data[key])
        else:
            for key in data:
                self.buffer[key].extend(data[key])

    def save(self, datapath=None):
        for key in self.buffer:
            path = datapath + f"{key}"
            np.save(file=path, arr=self.buffer[key])


if __name__ == '__main__':
    count = [0, 0]
    steps = []
    env = AMP(rew_coeff=rew_coeff_sou,
              # sense_noise='default',
              # control_name='forceThrustOmega'
              )
    ExpertAirDocking = Dataset()
    SuccessDocking = Dataset()
    for i in range(100):
        cache = Dataset(cache=True)  # 每轮对接数据缓存
        obs = env.reset()
        #time.sleep(0.1)
        for i in range(2000):

            # action1 = env.env.policy.stepThrustOmega(dynamics=env.env.dynamics, goal=env.env.goal)
            # action2 = env.env2.policy.stepThrustOmega(dynamics=env.env2.dynamics, goal=env.env2.goal)
            # action = np.concatenate([action1, action2])
            action = env.action_space.sample()

            next_obs, total_reward, done, info = env.step(action)
            mask = 1 - done
            data = {'obs': obs,
                    'action': action,
                    'total_reward': total_reward,
                    'next_obs': next_obs,
                    'mask': mask,
                    }
            cache.add(data)
            obs = next_obs
            if env.success_flag:
                steps.append(i)
            # time.sleep(0.03)
            if done:
                break
        if env.success_flag:
            count[0] += 1
            SuccessDocking.add(cache.buffer)
        count[1] += 1
        ExpertAirDocking.add(cache.buffer)
        print(count, steps[-1] if env.success_flag else None, np.mean(steps))



    data_file_path = "/dataset/ExpertAirDocking3000/test/"
    success_file_path = "/dataset/ExpertAirDocking3000/test/success/"
    ExpertAirDocking.save(data_file_path)
    SuccessDocking.save(success_file_path)
    # print(count)
    env.close()

    # test
    # load_expert_dataset = {}
    #
    # for key in ExpertAirDocking.buffer:
    #     load_data_file_path = os.path.join(data_file_path, key + '.npy')
    #
    #     load_expert_dataset[key] = np.load(load_data_file_path)
    #
    # a = 1












