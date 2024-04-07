import sys
sys.path.append("..")
import numpy as np
import gym
from gym import spaces
from env_air_sb3.Base import *
from collections import Counter
from controller.quad_utils import *

# try:
#     from garage.core import Serializable
# except:
#     print("WARNING: garage.core.Serializable is not found. Substituting with a dummy class")
#     class Serializable:
#         def __init__(self):
#             pass
#         def quick_init(self, locals_in):
#             pass

class AMP(gym.Env):

    """
    file_1:     : load the first urdf model
    file_2      : load the second urdf model
    reward_type : choose different reward
    rew_coeff   : reward super parameters
    train_type  : only train some one or two
    """

    def __init__(self,file_1="./rsc/AirDocking_6/air_docking_sou.urdf",
                file_2 = "./rsc/AirDocking_6/air_docking_sou.urdf" ,
                reward_type = "sou",
                rew_coeff = dict(),
                train_type = "double" ,
                sense_noise = None,
                control_name = "thrustOmega",
                max_step = 700,
                AquaML=False,
                dt = 0.005,
                change_goal_flag = True) -> None:
        
        self.AquaML = AquaML

        self.room_box = room_box
        self.obs_repr = obs_repr
        self.train_type = train_type
        self.control_name = control_name
        self.env = quadrotorEnv(file_1, name="F450", 
                                control_name=control_name, 
                                desired_force=desired_force,
                                reward_type=reward_type,
                                rew_coeff = rew_coeff,
                                sense_noise = sense_noise,
                                max_step = max_step,
                                dt = dt)
        self.env2 = quadrotorEnv(file_2, name="F450_2", 
                                first=False, world = self.env.world,
                                control_name=control_name,
                                desired_force=-desired_force,
                                reward_type=reward_type,
                                rew_coeff = rew_coeff,
                                sense_noise = sense_noise,
                                max_step = max_step,
                                dt=dt)
        self.env2.goal = goal2 
        self.env.quadrotor.setGeneralizedCoordinate(gc)
        self.env2.quadrotor.setGeneralizedCoordinate(gc2)
        self.rew_coeff = rew_coeff_sou
        self.reset()
        self.max_step = max_step
        self.action = np.zeros(8)
        self.action_prev = copy.deepcopy(self.action)
        
        self.rew_coeff.update(rew_coeff)
        # #pattern
        # self.change_goal_flag = change_goal_flag
        # self.dis_judge = 0
        #self.vel_ = 0
        self.last_dis_flag = False
        self.dt = dt
        
        

    def step(self, action=np.zeros(8)):
        self.action = action
        #print(self.env.step_count)
        #pdb.set_trace()
        if self.action_space.shape[0] == 8:
            env_action = action[:4]
            env2_action = action[4:]
        elif self.action_space.shape[0] == 36:
            env_action = action[:18]
            env2_action = action[18:]
        elif self.action_space.shape[0] == 6:
            env_action = action[:3]
            env2_action = action[3:]
        elif self.action_space.shape[0] == 3:
            env_action = self.env.goal
            env2_action = action
        # elif self.control_name == "velocityYaw":
        #     env_action = np.
        #     env2_action = action[:4]

        #self.getDesforce()
        #pdb.set_trace()
        #判定goal2是否还需要发生改变
        #self.change_goal()
        if self.train_type == "double":
            observation, reward1, done1, info1 = self.env.step(env_action)
            observation2, reward2, done2, info2 = self.env2.step(env2_action)
            self.env.world.integrate()
            
            critic_obs1 = self.env.get_critic_obs()
            critic_obs2 = self.env2.get_critic_obs()
            if done1 or done2:
                done = True
            else:
                done = False    
            #reward, info = self.getReward()
            if self.control_name == "velocityYaw":
                reward, info = self.contactRatio()
            else:
                reward, info = self.contactRatio()
            done = False
            # reward = reward1 + reward2
            # info = {key: info1.get(key, 0) + info2.get(key, 0) for key in set(info1) | set(info2)}
        elif self.train_type == "singleA":
            observation, reward, done, info = self.env.step(env_action)
            observation2, reward2, done2, info2 = self.env2.step(env2_action)
            
            critic_obs1 = self.env.get_critic_obs()
            critic_obs2 = self.env2.get_critic_obs()
            
            # critic_obs1 = np.concatenate([self.env.pos,])
            self.env.world.integrate()
        elif self.train_type == "singleB":
            observation2, reward, done, info = self.env2.step(env2_action)
            observation, reward2, done2, info2 = self.env.step(env_action)
            
            critic_obs1 = self.env.get_critic_obs()
            critic_obs2 = self.env2.get_critic_obs()
            self.env.world.integrate()
        #info = dict(Counter(info1) + Counter(info2)) #info1 + info2
        if self.control_name == "pidThrustOmega" and not done:
            if self.change_goal():
                done = self.find_pattern_done()
        else:
            done = (self.env.step_count > self.max_step)
        # if not done :
        #     done = self.find_pattern_done() or (self.env.step_count > self.max_step)
            # and self.env.step_count > self.max_step:
            # done = True
        # if self.control_name == "thrustOmega":
        #     self.change_goal_twice()
        #done = False
        #self.dis_judge = 0
        # if self.change_goal():
        done = self.find_pattern_done()
        if self.env.crashed or self.env2.crashed :
            done = True
          
        if self.control_name == "pidThrustOmega":
            self.action = np.concatenate([self.env.action, self.env2.action])
            
        amp_obs1 = deepcopy(observation)
        amp_obs2 = deepcopy(observation2)
        
        amp_obs1[:3] = amp_obs1[:3] + self.env.goal - self.goal1
        amp_obs2[:3] = amp_obs2[:3] + self.env2.goal - self.goal2
            
        obs = np.concatenate([amp_obs1,amp_obs2[:18]])
        
        
        critic_obs = np.concatenate([
            critic_obs1,
            critic_obs2[:18]
        ])
        
        # print("pos1: ", self.env.dynamics.pos)
        # print("pos2: ", self.env2.dynamics.pos)
        self.action_prev = copy.deepcopy(self.action)
        if self.AquaML:
            return obs, critic_obs, reward, done, {'rewards': info}
        else:
            return obs, reward, done, {'rewards': info}
            
    
    def get_observation_and_action_space(self,):
        if self.obs_repr == "xyz_vxyz_rot_omega_force_torque_xyz_vxyz_rot_omega":
            ## Creating observation space
            # pos, vel, rot, rot vel
            self.obs_comp_sizes = [3, 3, 9, 3, 3, 3 ,3, 3, 9, 3]
            self.obs_comp_names = ["xyz", "Vxyz", "R", "Omega", "force", "torque", "xyz2", "Vxyz2", "R2", "Omega2"]
            obs_dim = np.sum(self.obs_comp_sizes)
            obs_high =  np.ones(obs_dim)
            obs_low  = -np.ones(obs_dim)
            
            # xyz room constraints
            obs_high[0:3] = self.room_box[1] - self.room_box[0] #i.e. full room size
            obs_low[0:3]  = -obs_high[0:3]

            obs_high[24:27] = self.room_box[1] - self.room_box[0] #i.e. full room size
            obs_low[24:27]  = -obs_high[24:27]

            # Vxyz
            obs_high[3:6] = self.env.dynamics.vxyz_max * obs_high[3:6]
            obs_low[3:6]  = self.env.dynamics.vxyz_max * obs_low[3:6] 

            obs_high[27:30] = self.env.dynamics.vxyz_max * obs_high[27:30]
            obs_low[27:30]  = self.env.dynamics.vxyz_max * obs_low[27:30] 

            # Omega
            obs_high[15:18] = self.env.dynamics.omega_max * obs_high[15:18]
            obs_low[15:18]  = self.env.dynamics.omega_max * obs_low[15:18] 

            obs_high[39:42] = self.env.dynamics.omega_max * obs_high[39:42]
            obs_low[39:42]  = self.env.dynamics.omega_max * obs_low[39:42] 

        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
        #pdb.set_trace()
        if self.control_name == "thrustOmegaPolicy36":
            self.action_space = spaces.Box(np.concatenate([obs_low[:18],obs_low[-18:]]), np.concatenate([obs_high[:18],obs_high[-18:]]), dtype=np.float32)
        elif self.control_name == "thrustOmegaPolicy6":
            self.action_space = spaces.Box(np.concatenate([obs_low[:3],obs_low[:3]]), np.concatenate([obs_high[:3],obs_high[:3]]), dtype=np.float32)
        elif self.control_name == "thrustOmegaPolicy3":
            self.action_space = spaces.Box(obs_low[:3], obs_high[:3],dtype=np.float32)
        elif self.control_name == "velocityYaw":
            #pdb.set_trace()
            self.action_space = spaces.Box(np.concatenate([obs_low[3:6], [obs_low[17]]]), np.concatenate([obs_high[3:6], [obs_high[17]]]), dtype=np.float32)
        else:
            self.action_space = None
        return self.observation_space, self.action_space

    def get_action_space(self,):
        circle_per_sec = 2 * np.pi
        max_rp =  circle_per_sec
        max_yaw = circle_per_sec
        min_g = -1.0
        max_g = self.env.dynamics.thrust_to_weight - 1.0
        self.low  = np.array([min_g, -max_rp, -max_rp, -max_yaw, min_g, -max_rp, -max_rp, -max_yaw])
        self.high = np.array([max_g,  max_rp,  max_rp,  max_yaw, max_g, max_rp,  max_rp,  max_yaw])
        return spaces.Box(self.low, self.high, dtype=np.float32)

    #改变目标位置的判定
    def change_goal(self,):
        if self.env.step_count > 1000 and not np.any(self.env.dynamics.contact_force) and self.change_goal_flag:
            self.vel_ += 0.001
            #self.vel_ += 100.1
            self.env2.goal[1] += self.vel_ * self.env.dt
        if ((np.abs(self.env2.dynamics.pos[2] - self.env.dynamics.pos[2])>0.03) \
            and np.any(self.env.dynamics.contact_force) \
            and np.abs(self.env2.goal[1] - self.env.goal[1]) <1.2 ) :#\
            #or ( self.env.step_count > 3000):#and self.change_goal_flag:
            #pdb.set_trace()
            #print("卡住")
            self.vel_ = -0.3
            self.env2.goal[1] += self.vel_ * self.env.dt
            #print(f"self.env2.goal is {self.env2.goal[1]} ")
            self.vel_ = 0.3
            return False
            
        if np.abs(self.env2.goal[1] - self.env.goal[1]) <= 0.54:
            self.change_goal_flag = False  
        else:
            self.change_goal_flag = True  
        return True

    #改变目标位置的判定
    def change_goal_twice(self,):
        pass
        # if np.linalg.norm(self.env2.dynamics.pos[1] - self.env2.goal[1]) <= 0.1:
        #     if np.abs(self.env.goal[1] - self.env2.goal[1]) == 1.:
        #         self.env2.goal[1] -= 0.24
        #     elif np.abs(self.env.goal[1] - self.env2.goal[1]) == 1.-0.24:
        #         self.env2.goal[1] -= 0.36

    #对接成功的判定
    def find_pattern_done(self,):
        # dis = self.env.dynamics.pos - self.env2.dynamics.pos
        # dis_ = np.sum(np.square(dis))
        # angle = rotationMatrixToEulerAngles(self.env.rot_b)*(180/np.pi)
        # if dis_ <= 0.54 and angle[0] < 8:# and self.last_dis_flag:
        #     self.dis_judge += 1
        # else:
        #     self.dis_judge = 1
        if self.dis_judge > 150:
            # pdb.set_trace()
            # print("完全对接")
            return True
        else:
            return False
        # return False

    """
    reward设计考虑两个一起考虑或者一个来考虑之类的, 回头慢慢尝试
    """
    def getReward(self, done, done1, done2, reward1, reward2, info1, info2):
        loss_done = rew_coeff["done"] * int(done)
        
        #需要近距离移动
        goal_y_loss = rew_coeff["goal_y"] * np.linalg.norm(self.env.dynamics.pos[1] - self.env.dynamics.pos[1] - 0.54)
        goal_xz_loss = rew_coeff["goal_xz"] * (np.linalg.norm(self.env.goal[0] - self.env2.goal[0]) + np.linalg.norm(self.env.goal[2] - self.env2.goal[2]))
        
        #判定对接
        dis = self.env.dynamics.pos - self.env2.dynamics.pos
        dis_ = np.sum(np.square(dis))
        angle = rotationMatrixToEulerAngles(self.env.rot_b)*(180/np.pi)
        if dis_ <= 0.54 and angle[0] < 8:# and self.last_dis_flag:
            self.dis_judge += 1
        else:
            self.dis_judge = 1
        if self.dis_judge > 200:
            print("完全对接")
        airdocking_loss = rew_coeff["airdocking"] * self.dis_judge
        
        reward = -self.env.step_count/100 * np.sum([
            loss_done,
            goal_y_loss,
            goal_xz_loss,
            airdocking_loss,
            reward1,
            reward2
        ])

        rew_info = {
            "rew_main":reward,
            "rew_pos1":info1["rew_pos"],
            'rew_force1': info1["rew_force"],
            "rew_spin1": info1["rew_spin"],
            "rew_spin_z1": info1["rew_spin_z"],
            "rew_omega1": info1["rew_omega"],
            "rew_crash1": info1["rew_crash"],
            "rew_pos2":info2["rew_pos"],
            'rew_force2': info2["rew_force"],
            "rew_spin2": info2["rew_spin"],
            "rew_spin_z2": info2["rew_spin_z"],
            "rew_omega2": info2["rew_omega"],
            "rew_crash2": info2["rew_crash"],
            "rew_done": -loss_done,
            "goal_y_loss": -goal_y_loss,
            "goal_xz_loss": -goal_xz_loss,
            "airdocking_loss": -airdocking_loss
        }
        if np.isnan(reward) or not np.isfinite(reward):
            for key, value in locals().items():
                print('%s: %s \n' % (key, str(value)))
            raise ValueError('QuadEnv: reward is Nan')
        return reward, rew_info
        # if self.dis_judge > 200:
        #     #print("完全对接")
        #     return self.dis_judges
        # else:

    def getLossYXZ(self, position1, position2):
        loss_y = self.rew_coeff["goal_y"] * np.linalg.norm(position1[1] - position2[1] - 0.54)
        
        x_z_env1 = np.array([position1[0], position1[2]])
        x_z_env2 = np.array([position2[0], position2[2]])
        
        dis_xz = np.linalg.norm(x_z_env1-x_z_env2)
        
        loss_xz = self.rew_coeff["goal_xz"]*dis_xz
        
        return loss_y, loss_xz, dis_xz

    def getRewardRatio(self,):
        reward , info = self.contactRatio()
        force_rew = -self.rew_coeff["force"] * np.linalg.norm(self.env.dynamics.contact_force - self.env.dynamics.desired_force_world)
        
        
        action_rew = -self.rew_coeff["action_change"] * np.linalg.norm(self.action_prev - self.action)

        reward = np.sum([reward,
                        force_rew,
                        action_rew])

        info["force_rew"] = force_rew
        info["action_rew"] = action_rew

        return reward, info

    def contactRatio(self,):
        #无人机当前位置转机体位置描述：
        # body_position1 = self.env.dynamics.rot @ self.env.dynamics.position 
        # body_position2 = self.env2.dynamics.rot @ self.env2.dynamics.position 
        
        # 1 飞机是桶，2 飞机是尖头
        # 强制2号飞机在1号飞机右边
        
        # TODO:矫正物理参数
        tau1_xyz = np.array([0,-0.45,-0.03])
        tau2_xyz = np.array([0,0.39,-0.03])
        
        depth_tau = self.rew_coeff["depth_tau"] # 套筒深度
        
        diameter = 0.10 #套筒直径
        
        # 套筒goal位置
        goal1 = self.env.goal +  tau1_xyz
        goal2 = self.env.goal + np.array([0,-0.31,-0.03])
        
        # reward设置
        maximun_overlap_reward = self.rew_coeff["maximun_overlap_reward"]
        minimun_overlap_reward = self.rew_coeff["minimun_overlap_reward"]
        minimun_joint_depth = self.rew_coeff["minimun_joint_depth"] # 最小接触深度 p2-p1, 启动接触奖励最小值 
        
        
        # coeficient
        overlap_coeff = self.rew_coeff["overlap_coeff"]
        position_coeff = self.rew_coeff["position_coeff"]
        orientation_coeff = self.rew_coeff["orientation_coeff"]
        
        
        
        ##############################
        # 获取套筒端位置, 位置姿态位置
        ##############################
        
        # 获取两个无人机位置
        position1 = self.env.dynamics.pos
        position2 = self.env2.dynamics.pos
        
        # 获取套筒在世界坐标系下的表示
        tau1_xyz_world = position1 + self.env.dynamics.rot @ tau1_xyz
        tau2_xyz_world = position2 + self.env2.dynamics.rot @ tau2_xyz
        
        # 套筒和套尖端loss
        loss_pos1 = np.linalg.norm(goal1 - tau1_xyz_world)
        loss_pos2 = np.linalg.norm(goal2 - tau2_xyz_world)
        
        # 子机姿态loss
        rot_cos1 = ((self.env.dynamics.rot[0, 0] + self.env.dynamics.rot[1, 1] + self.env.dynamics.rot[
            2, 2]) - 1.) / 2.
        rot_cos2 = ((self.env2.dynamics.rot[0, 0] + self.env2.dynamics.rot[1, 1] + self.env2.dynamics.rot[
            2, 2]) - 1.) / 2.
        
        loss_rotaion1 = np.arccos(np.clip(rot_cos1, -1., 1.))
        loss_rotaion2 = np.arccos(np.clip(rot_cos2, -1., 1.))
        
        loss_pos = (loss_pos1 + loss_pos2) * position_coeff
        loss_rotaion = (loss_rotaion1 + loss_rotaion2) * orientation_coeff
        
        ##############################
        # 判断是否接触，给予趋势奖励
        ##############################
        
        # 接触最大奖励距离
        maximun_overlap_reward_dis = minimun_joint_depth + depth_tau
        
        # 2号无人机尖头位置到1号无人机套筒右端圆心位置的距离(x,z)
        p1_xz = np.array([tau1_xyz_world[0], tau1_xyz_world[2]])
        p2_xz = np.array([tau2_xyz_world[0], tau2_xyz_world[2]])
        
        p1_y = tau1_xyz_world[1]
        p2_y = tau2_xyz_world[1]
        
        dis_xz = np.linalg.norm(p1_xz-p2_xz)
        
        # TODO: reward提升，但是位置下降原因：不是绝对位置上的对接而是相对位置上的对接
        
        if not self.env.crashed or self.env2.crashed:
            if dis_xz <= diameter/2.0 and p2_y - p1_y <= depth_tau+0.025:
            # print(f"p1_y-p2_y is : {p1_y-p2_y}")
                if p1_y - p2_y <= minimun_joint_depth:
                    
                    overlapping_reward = minimun_overlap_reward - (p1_y - p2_y - minimun_joint_depth) * (maximun_overlap_reward - minimun_overlap_reward) / (maximun_overlap_reward_dis)
                    overlapping_reward = overlapping_reward * overlap_coeff 
                #pdb.set_trace()
                    if p2_y - p1_y >= depth_tau-0.025:
                        # cumulate jis_judge
                        self.dis_judge += 1 
                        # if self.dis_judge > 200:
                        # #     # print("接触成功")
                        #     force = self.env.dynamics.contact_force - self.env.dynamics.desired_force_world
                        #     print(f"force is {self.env.dynamics.contact_force[1]}")
                        
                        # if self.dis_judge >= 200:
                        #     print("接触成功")
                else:
                    overlapping_reward = 0
            else:
                overlapping_reward = 0
        else:
            overlapping_reward = 0
        
        # if dis_xz <= diameter/2.0:
        #     # print(f"p1_y-p2_y is : {p1_y-p2_y}")
        #     if p1_y - p2_y <= minimun_joint_depth:
                
        #         overlapping_reward = minimun_overlap_reward - (p1_y - p2_y - minimun_joint_depth) * (maximun_overlap_reward - minimun_overlap_reward) / (maximun_overlap_reward_dis)
        #         overlapping_reward = overlapping_reward * overlap_coeff
        #         #pdb.set_trace()
        #         if p2_y - p1_y >= depth_tau-0.025:
                    
        #             print("接触成功")
        #     else:
        #         overlapping_reward = 0
        # else:
        #     overlapping_reward = 0

        force = self.env.dynamics.contact_force - self.env.dynamics.desired_force_world
        force_rew = -self.rew_coeff["force"] * abs(force[1])
        
        
        reward = -self.dt*np.sum([
            loss_pos,
            loss_rotaion,
            -overlapping_reward,
            force_rew
        ])
        
        # print(reward)
        
        rew_info = {
            "rew_main":reward,
            "loss_pos1": loss_pos1,
            "loss_pos2": loss_pos2,
            "loss_pos": loss_pos,
            "loss_rotaion1": loss_rotaion1,
            "loss_rotaion2": loss_rotaion2,
            "loss_rotaion": loss_rotaion,
            "overlapping_reward": overlapping_reward,
            'force_rew': force_rew,
        }
        

        return reward, rew_info
        
    def setGoal2(self,):
        tau1_xyz = np.array([0,-0.45,-0.03])
        tau2_xyz = np.array([0,0.39,-0.03])
        
        depth_tau = self.rew_coeff["depth_tau"] # 套筒深度
        
        diameter = 0.10 #套筒直径
        
        # 套筒goal位置
        goal1 = self.env.goal +  tau1_xyz
        goal2 = self.env.goal + np.array([0,-0.31,-0.03])
        uav_goal2 = goal2 - np.array([0, 0.39, -0.03])

        return uav_goal2

    def getReward(self,):
        #loss_done = rew_coeff["done"] * int(done)
        
        #需要近距离移动
        # TODO: 存在俩种情况
        if self.control_name == "thrustOmegaPolicy3":
            goal_y_loss, goal_xz_loss, dis_xz = self.getLossYXZ(self.env.goal, self.env2.goal)
        elif self.control_name == "thrustOmega":
            goal_y_loss, goal_xz_loss, dis_xz = self.getLossYXZ(self.env.dynamics.position, self.env2.dynamics.position)
        else:
            goal_y_loss = self.rew_coeff["goal_y"] * np.linalg.norm(self.env2.dynamics.pos[1] - self.env.dynamics.pos[1] - 0.67)
            
            x_z_env1 = np.array([self.env.dynamics.pos[0],self.env.dynamics.pos[2]])
            x_z_env2 = np.array([self.env2.dynamics.pos[0],self.env2.dynamics.pos[2]])
            
            dis_xz = np.linalg.norm(x_z_env1-x_z_env2)
            
            goal_xz_loss = self.rew_coeff["goal_xz"]*dis_xz
        
        # TODO: 更换成2范数，xz reward过大
        # if self.control_name == "thrustOmegaPolicy3":
            
        #     goal_xz_loss = self.rew_coeff["goal_xz"] * (np.linalg.norm(self.env.goal[0] - self.env2.goal[0]) + np.linalg.norm(self.env.goal[2] - self.env2.goal[2]))
        # elif self.control_name == "thrustOmega":
        #     goal_xz_loss = self.rew_coeff["goal_xz"] * (np.linalg.norm(self.env.dynamics.pos[0] - self.env.dynamics.pos[0]) + np.linalg.norm(self.env.dynamics.pos[2] - self.env.dynamics.pos[2]))
        
        #判定对接
        dis = self.env.dynamics.pos - self.env2.dynamics.pos
        dis_ = np.sum(np.square(dis))
        angle = rotationMatrixToEulerAngles(self.env.rot_b)*(180/np.pi)
        
        if dis_ <= 0.54 and angle[0] < 8 and dis_xz<0.04:# and self.last_dis_flag:
            self.dis_judge = 1.0 # encourage
            print("对接")
        else:
            self.dis_judge = 0.0
        rew_airdocking_loss = -self.rew_coeff["airdocking"] * self.dis_judge
        
        if self.env.crashed or self.env2.crashed :
            crash_loss = self.rew_coeff["crash"] * 1.0
        else:
            crash_loss = 0.

        if (np.abs(self.env2.dynamics.pos[2] - self.env.dynamics.pos[2])>0.03) \
            and np.any(self.env.dynamics.contact_force) :
            ka_loss = self.rew_coeff["ka"] * np.abs(self.env2.dynamics.pos[2] - self.env.dynamics.pos[2]) 
            # 试试恒定调试 TODO
        else:
            ka_loss = 0.
        reward = -self.dt * np.sum([
            goal_y_loss,
            goal_xz_loss,
            rew_airdocking_loss,
            crash_loss,
            ka_loss
        ])

        rew_info = {
            "rew_main":reward,
            "goal_y_rew": -goal_y_loss,
            "goal_xz_rew": -goal_xz_loss,
            "airdocking_rew": -rew_airdocking_loss,
            "crash_rew": -crash_loss,
            "ka_rew" : -ka_loss
        }
        if np.isnan(reward) or not np.isfinite(reward):
            for key, value in locals().items():
                print('%s: %s \n' % (key, str(value)))
            raise ValueError('QuadEnv: reward is Nan')
        return reward, rew_info
        

    def set_init_random_goal(self,):
        tmp_xy = np.random.randint(-10, 9, (2))
        tmp_z = np.random.randint(6,8,(1))
        tmp = np.insert(tmp_xy, 2, values = tmp_z, axis=0)
        point = np.random.rand(3) #多goal （0，0，2）(3，4，2)（6，0，2）
        goal = tmp + point
        pos = np.random.uniform(-2, 2, size=(3,)) + goal
        pos = np.clip(pos, a_min=room_box[0], a_max=room_box[1])
        return goal, pos
    
    def set_init_fixed_goal(self,):
        goal = np.array([0.,0.,3.])
        pos = np.random.uniform(-0.4, 0.4, size=(3,)) + goal
        return goal, pos

    def reset(self,):
        #pdb.set_trace()
        #random goal and pos
        
        # goal, pos = self.set_init_random_goal()
        goal, pos = self.set_init_fixed_goal()
        
        # TODO: 使用fixed goal
        pos = goal
        gc[:3] = pos
        
        # goal2 = copy.deepcopy(goal)
        # goal2[1] -= 1.0# 2.0
        # pos2 = copy.deepcopy(pos)
        # pos2[1] -=  1.0# 3.0
        # gc2[:3] = pos2
        
        pos2 = np.zeros(3)

        goal2 = self.setGoal2()
        pos2[1] = goal2[1] + random.uniform(-0.5, -0.3)
        pos2[0] = goal2[0] + random.uniform(-0.15, 0.15)
        pos2[2] = goal2[2] + random.uniform(-0.15, 0.15)
        gc2[:3] = pos2

        self.env.goal = goal
        self.env2.goal = goal2
        
        self.goal1 = deepcopy(goal)
        self.goal2 = deepcopy(goal2)
        
        self.env.quadrotor.setGeneralizedCoordinate(gc)
        self.env2.quadrotor.setGeneralizedCoordinate(gc2)

        if self.control_name == "thrustOmegaPolicy3":
            #random_att
            # rot1 = randomRot(0, np.pi/20)
            # self.env.quadrotor.setBaseOrientation(rot1)

            rot2 = randomRot(0, np.pi/40)
            self.env.quadrotor.setBaseOrientation(rot2)
        else:
            #random_att
            # rot1 = randomRot(0, np.pi/20)
            # self.env.quadrotor.setBaseOrientation(rot1)

            rot2 = randomRot(0, np.pi/40)
            self.env.quadrotor.setBaseOrientation(rot2)
        

        #pdb.set_trace()
        observation = self.env.reset()
        #self.env2.world = self.env.world
        observation2 = self.env2.reset()
        #self.env.startServer()
        
        #完全对接的判定
        self.vel_ = 0
        self.change_goal_flag = False
        self.dis_judge = 0
        
        critic_obs1 = self.env.get_critic_obs()
        critic_obs2 = self.env2.get_critic_obs()
        
        critic_obs = np.concatenate([
            critic_obs1,
            critic_obs2[:18]
        ])
        
        obs = np.concatenate([observation,observation2[:18]])

        if self.control_name.startswith("thrustOmegaPolicy"):
            self.observation_space, self.action_space = self.get_observation_and_action_space()
        else:
        #elif self.control_name == "thrustOmega":
            self.observation_space, self.action_space = self.get_observation_and_action_space()
            self.action_space = self.get_action_space()
        self.env.world.integrate()
        if self.AquaML:
            return obs, critic_obs
        else:
            return obs

    def close(self,):
        self.env.close()
        #self.env2.close()


from gym.envs.registration import register

