import os
import copy
import numpy as np
import raisimpy as raisim
import gym
from gym import spaces

#import articulated_system
import time


import sys
sys.path.append("/home/ming/aaa/quad_raisim")
from controller.quadrotor_control import *
from controller.thrust_omega_control import *
from controller.quad_utils import *
from controller.sensor_noise import *
from controller.thrust_omega_control import *

from env_air_sb3.env_params import *
from env_air_sb3.reward import *

"""
噪声的加入放入了Dynamic中
"""
class quadrotorEnv():
    def __init__(self, file_="/home/fyt/project/quad_raisim/rsc/F450_1009/F450.urdf", sense_noise = None, 
                 name = "F450", control_name = None,first = True, world= None, others={},
                 desired_force=np.zeros(3),
                 reward_type="sou",
                 rew_coeff=dict(),
                 max_step=700,
                 dt=0.005,
                 model_file = "/home/fyt/project/quad_raisim/models/thrust_omega/seed_001/network_evaluate.so"  # TODO
                ):
        raisim.World.setLicenseFile("/home/ming/.raisim/activation.raisim")
        self.server = False
        self.goal = goal
        self.quadrotor_urdf_file = file_
        self.dt = dt
        self.name = name
        self.world = world
        self.step_count = 0
        self.first = first
        self.others = others
        self.close_flag = False
        self.control_name = control_name
        self.desired_force = desired_force
        self.reward_type = reward_type
        self.rew_coeff = rew_coeff
        self.action = np.zeros(4)
        self.action_prev = np.zeros(4)
        self.crashed = False
        self.sense_noise = sense_noise
        self.max_step = max_step

        self.force_torque = np.zeros(6)
        self.model_file = model_file

        
        
        # self.rew_coeff =  {
        #     "pos": 1., "pos_offset": 0.1, "pos_log_weight": 1., "pos_linear_weight": 0.1,
        #     "effort": 0.01,
        #     "crash": 1.,
        #     "orient": 1., "yaw": 0., "rot": 0., "attitude": 0.,
        #     "spin_z": 0.5, "spin_xy": 0.,
        #     "spin": 0.,
        #     "vel": 0.,
        #     "omega": 0.,
        #     "action_change":0.,
        #     "alpha_a":0.,
        #     "alpha_v":0.,
        #     "alpha_r":0.,
        #     "action_mean":0.,
        #     "thrust_change":0.,
        #     "thrust_mean":0.,
        #     "one_p":0.,
        #     }
        self.startServer()
        self.reset()
        
    
    def startServer(self,): 
        if self.first:
            self.world = raisim.World()
            self.world.setTimeStep(self.dt) #设置模拟器的固定时间步长
            self.server = raisim.RaisimServer(self.world)
            self.world.addGround(material="steel")
            self.server.launchServer(8088)
            self.quadrotor = self.world.addArticulatedSystem(self.quadrotor_urdf_file)
            self.server.focusOn(self.quadrotor)
        else:
            self.quadrotor = self.world.addArticulatedSystem(self.quadrotor_urdf_file)
        self.quadrotor.setName(self.name)
        

    def setQuadrotor(self, others):
        friction = random.uniform(0.1, 0.9)
        restitution = random.uniform(0.1, 0.7)
        #self.world.setMaterialPairProp("2Al2", "2Al2", friction, restitution, 0.001)
        self.footFrameIndex = self.quadrotor.getFrameIdxByName("body_joint")
        prop_index1 = self.quadrotor.getFrameIdxByName("prop_joint1")
        prop_index2 = self.quadrotor.getFrameIdxByName("prop_joint2")
        prop_index3 = self.quadrotor.getFrameIdxByName("prop_joint3")
        prop_index4 = self.quadrotor.getFrameIdxByName("prop_joint4")
        self.bodyidx = self.quadrotor.getBodyIdx("base_link")
        self.body_frame = self.quadrotor.Frame.BODY_FRAME
        self.world_frame = self.quadrotor.Frame.WORLD_FRAME
        #pdb.set_trace()
        self.mass = sum(self.quadrotor.getMass())
        #self.inertia = self.quadrotor.setInertia([[0.39990040143520006, 0.11735786261519998, 0.47971748234000006],
                                                    # [0.39990040143520006, 0.11735786261519998, 0.47971748234000006],
                                                    # [0.39990040143520006, 0.11735786261519998, 0.47971748234000006],
                                                    # [0.39990040143520006, 0.11735786261519998, 0.47971748234000006],
                                                    # [0.39990040143520006, 0.11735786261519998, 0.47971748234000006]])
        self.inertia = inertia#np.diagonal(sum(self.quadrotor.getInertia())) # inertia 
        #pdb.set_trace()
        self.dynamics = Dynamics(mass=self.mass, inertia=inertia, sense_noise=self.sense_noise, desired_force=self.desired_force)
        if self.control_name == "thrustOmega" or self.control_name == "forceThrustOmega":
            self.control = OmegaThrustControl(self.dynamics)
            
            # self.action_space = self.control.action_space(dynamics)
            # self.observation_space = self.get_observation_space()
        elif self.control_name == "NonPos":
            self.control = NonlinearPositionController(self.dynamics, force_control=force_control)
        elif self.control_name == "pidThrustOmega":
            self.policy = NonlinearPositionController(self.dynamics, force_control=force_control)
            self.control = OmegaThrustControl(self.dynamics)
        elif self.control_name == "thrustOmegaPolicy6": # output is change goal
            #self.goal = 
            self.control = NonlinearPositionController(self.dynamics, force_control=force_control)
        elif self.control_name == "thrustOmegaPolicy3": # output is change goal
            #self.goal = 
            self.action = np.zeros(3)
            self.action_prev = np.zeros(3)
            self.control = NonlinearPositionController(self.dynamics, force_control=force_control)
        elif self.control_name == "velocityYaw": # output is change goal
            #self.goal = 
            self.action = np.zeros(4)
            self.action_prev = np.zeros(4)
            self.control = VelocityYawControl(self.dynamics)
        elif self.control_name.startswith("thrustOmegaPolicy"):
            self.policy = controlThrustOmega(self.model_file)
            self.control = OmegaThrustControl(self.dynamics)
        

        self.prop_index = np.array([prop_index1, prop_index2, prop_index3, prop_index4])
        self.control_flag = control_flag
        prop_vel = 20
        gv = [0, 0, 0,  0.0,  0.0, 0, prop_vel,prop_vel,prop_vel,prop_vel]
        self.quadrotor.setGeneralizedVelocity(gv)
        self.others = others
        #self.reset()

    #后续可添加noise
    def state(self, ):
        self.position_w = self.quadrotor.getPosition(0)
        self.lineVel_w = self.quadrotor.getVelocity(0)
        angVel_w = self.quadrotor.getAngularVelocity(self.bodyidx) #world 
        #pdb.set_trace()
        quan = self.quadrotor.getBaseOrientation()
        self.rot_b = eulerAnglesToRotationMatrix(quan2angle(quan))
        #self.rot_b = self.quadrotor.getFrameOrientation(self.footFrameIndex)  #world坐标系下姿态的表示 
        self.angVel_b = np.array(self.rot_b @ np.mat(angVel_w).T)[:,0]
        self.getLinearAcc()
        self.position_w,self.lineVel_w,self.rot_b,self.angVel_b,self.acc = self.dynamics.sense_noise.add_noise(self.position_w,
                                            self.lineVel_w,
                                            self.rot_b,
                                            self.angVel_b,
                                            self.acc,
                                            self.dt)
        self.dynamics.update_state(self.position_w, self.lineVel_w, self.angVel_b, self.rot_b)

    def getLinearAcc(self,):
        self.acc = (self.lineVel_w - self.last_vel) / self.dt
        self.last_vel = copy.deepcopy(self.lineVel_w)
        self.accelerometer = np.matmul(self.rot_b.T, self.acc + [0, 0, GRAV])

    def step(self, action=None):
        self.action = action
        
        if self.control_name == "thrustOmegaPolicy36":
            self.action = self.policy.step(action)
        elif self.control_name == "thrustOmegaPolicy6":
            self.goal = copy.deepcopy(action)
            thrust, torque = self.control.step(dynamics=self.dynamics, goal=self.goal, action = None, dt=self.dt, flag=self.control_flag)
        elif self.control_name == "thrustOmegaPolicy3":
            self.goal = copy.deepcopy(action)
            thrust, torque = self.control.step(dynamics=self.dynamics, goal=self.goal, action = None, dt=self.dt, flag=self.control_flag)
        elif self.control_name == "forceThrustOmega":
            thrust, torque = self.control.stepForce2(dynamics=self.dynamics, goal=self.goal, action = self.action, dt=self.dt, flag=self.control_flag)
        #动力学更新
        #pdb.set_trace()
        elif self.control_name == "pidThrustOmega":
            self.action = self.policy.stepThrustOmega(dynamics=self.dynamics, goal=self.goal, action = self.action, dt=self.dt, flag=self.control_flag)
            thrust, torque = self.control.step(dynamics=self.dynamics, goal=self.goal, action = self.action, dt=self.dt, flag=self.control_flag)
        # elif self.control_name == "velocityYaw":
        #     #pdb.set_trace()
        #     #self.goal = copy.deepcopy(action)
        #     thrust, torque = self.control.step(dynamics=self.dynamics, goal=self.goal, action = self.action, dt=self.dt, flag=self.control_flag)
        else:
            thrust, torque = self.control.step(dynamics=self.dynamics, goal=self.goal, action = self.action, dt=self.dt, flag=self.control_flag)
        #print(thrust,torque)
        for i in range(len(thrust)):
            self.thrust = thrust[i]
            self.torque = torque[i]
            #仿真环境更新
            self.step_()
            
        #状态更新
        self.state()

        #外力获取
        self.force_torque = self.contact_check_contact()
        self.dynamics.contact_force = self.force_torque[:3]
        self.dynamics.contact_torque = self.force_torque[3:]
        #self.getDesforce()
        #pdb.set_trace()
        self.dynamics.desired_force_world = self.dynamics.rot @ self.dynamics.desired_force
        if False in np.concatenate([self.dynamics.pos < room_box[1], room_box[0] < self.dynamics.pos]):
            #pdb.set_trace()
            self.crashed = True
        else:
            self.crashed = False
        #pdb.set_trace()
        reward, info = self.getReward()
        done = False
        
        if self.step_count > self.max_step or self.find_crash() or self.find_pattern_Done():
            done = True
            #print(f"self.name is {self.name}, self.dynamics.pos is {self.dynamics.pos}")
            #print("done 1")
        # elif self.step_count > 500 and self.position_w[2] < 0.3:
        #     done = True
        #     print("done 2")
        else:
            done = False
        #print(f"step is {self.step_count}, done is {done}")
        #飞机状态
        observation = np.concatenate([self.position_w - self.goal, 
                                      self.lineVel_w,
                                      self.rot_b.flatten(),
                                      self.angVel_b,
                                      self.force_torque])
        self.observation = observation
        self.step_count += 1
        self.action_prev = copy.deepcopy(self.action)
        return observation, reward, done, info
    
    def get_critic_obs(self,):
        
        critic_obs = np.concatenate([self.dynamics.pos - self.goal, 
                                      self.dynamics.vel,
                                      self.dynamics.rot.flatten(),
                                      self.dynamics.omega,
                                      self.force_torque])
        return critic_obs

    def find_crash(self,):
        if self.position_w[2] < -0.5 :
            #print("position_w z done")
            return True
        else:
            return False
        # if self.step_count > self.max_step/3 and not np.any(self.env.dynamics.contact_force):
        #     return True
            
    def find_pattern_Done(self,):
        cur_angle = rotationMatrixToEulerAngles(self.rot_b)*(180/np.pi)
        if cur_angle[0] > 4 or cur_angle[1] > 4 :
            self.cur_angle_limit += 1
            if self.cur_angle_limit > 40:
                #print("---cur_angle_limit---")
                return True
        if np.abs(self.position_w[1] - self.goal[1]) > 1.0:
            self.pos_y_limit += 1
            if self.pos_y_limit > 20:
                #print("---pos_y_limit---")
                return True
        if self.angVel_b[0] > 0.3:
            self.angle_b_limit += 1
            if self.angle_b_limit/self.max_step > 0.1:
                #print("---angle_b_limit---")
                return True
        if np.abs(self.lineVel_w[1]) > 0.42:
            self.vel_limit += 1
            if self.vel_limit > self.max_step * 0.3:
                #print("---vel_limit---")
                return True
        else:
            return False
           

    def step_(self, ):
        self.dynamics.update_state(self.position_w, self.lineVel_w, self.angVel_b, self.rot_b)
        if self.control_flag == "body":
            #pdb.set_trace()
            thrust_world = np.array(self.rot_b @ np.mat([0.,0., self.thrust]).T)[:,0]
            #thrust_world = np.array([0.,0.,thrust])
            self.sim_body_step(thrust_world, self.torque)
        elif self.control_flag == "prop":
            final_thrust = []
            for j in range(len(self.thrust)): 
                tmp = np.array(self.rot_b @ np.mat([0.,0.,self.thrust[j]]).T)[:,0]
                final_thrust.append(tmp)
            self.thrust = [[0.,0.,self.thrust[0]], [0.,0.,self.thrust[1]], [0.,0.,self.thrust[2]], [0.,0.,self.thrust[3]]]
            self.sim_prop_step(final_thrust , self.torque)
    
    def getReward(self,):
        #pdb.set_trace()
        #return 0, {}
        if self.reward_type == "sou":
            reward, rew_info = compute_reward_weighted_sou(self.dynamics, self.goal, self.action, self.dt, self.crashed, 
                            rew_coeff=self.rew_coeff, action_prev=self.action_prev)
        elif self.reward_type == "force_track":
            reward, rew_info = compute_reward_force_track(self.dynamics, self.goal, self.action, self.dt, self.crashed, 
                            rew_coeff=self.rew_coeff, action_prev=self.action_prev)
        elif self.reward_type == "sou_force_track":
            reward, rew_info = compute_reward_weighted_force_track(self.dynamics, self.goal, self.action, self.dt, self.crashed, 
                            rew_coeff=self.rew_coeff, action_prev=self.action_prev)
        elif self.reward_type == "None":
            reward, rew_info = 0., {}
        return reward, rew_info

    def contact_check_contact(self,):
        forces = []
        torques = []
        if self.position_w[2] > 0.5:
            for contact in self.quadrotor.getContacts():
                if self.bodyidx == contact.getlocalBodyIndex():
                    # 三维力的获取
                    force = (contact.getContactFrame().transpose() @ contact.getImpulse().reshape(-1,1)).reshape(1,-1)[0] /self.dt
                    if not contact.isObjectA():
                        force = list(-np.array(force))
                    forces.append(force)

                    #三维力矩的获取
                    torque_pos = contact.get_position()
                    moment_arm = torque_pos - self.position_w
                    torque = np.cross(moment_arm, force)
                    torques.append(torque)       
        if len(forces) == 0:
            return np.zeros(6)
        else:
            #pdb.set_trace()
            res = list(map(sum, zip(*forces)))
            res.extend(list(map(sum, zip(*torques))))
            #res = res/self.dt
            return np.array(res)
        
    #加在整个body上
    def sim_body_step(self, thrust, torque, force_pos=np.zeros(3)):
        #pdb.set_trace()
        #but thrust need change
        self.quadrotor.setExternalTorque(self.bodyidx, torque)
        self.quadrotor.setExternalForce(self.bodyidx, self.world_frame, thrust, self.body_frame, force_pos)
        #self.quadrotor.setExternalForce(self.bodyidx, self.body_frame, thrust, self.world_frame, self.position_w)

    #加在四个旋翼上
    def sim_prop_step(self, thrust, torque):
        pos1 = self.prop_tuning(0)
        pos2 = self.prop_tuning(1)
        pos3 = self.prop_tuning(2)
        pos4 = self.prop_tuning(3)
        self.quadrotor.setExternalTorque(self.bodyidx, torque)
        self.quadrotor.setExternalForce(self.bodyidx, self.body_frame, np.array(thrust[0]), self.world_frame, pos1)
        self.quadrotor.setExternalForce(self.bodyidx, self.body_frame, np.array(thrust[1]), self.world_frame, pos2)
        self.quadrotor.setExternalForce(self.bodyidx, self.body_frame, np.array(thrust[2]), self.world_frame, pos3)
        self.quadrotor.setExternalForce(self.bodyidx, self.body_frame, np.array(thrust[3]), self.world_frame, pos4)

    def prop_tuning(self, index):
        prop_pos = self.quadrotor.getPosition(self.prop_index[index]) #+ pos_w
        return prop_pos

    def reset(self,):
        #pattern
        self.cur_angle_limit = 0
        self.pos_y_limit = 0
        self.angle_b_limit = 0
        self.vel_limit = 0
        
        self.last_vel = 0.
        self.setQuadrotor(self.others)
        self.state()
        self.dynamics.update_state(self.position_w, self.lineVel_w, self.angVel_b, self.rot_b)
#        self.dynamics.reset()  # TODO
        self.step_count = 1.
        
        if self.control_name == "pidThrustOmega":
            self.policy.action = np.zeros_like(self.policy.action)
        
        return np.concatenate([self.position_w-self.goal, self.lineVel_w, self.rot_b.flatten(), self.angVel_b,np.zeros(6) ])

    def close(self,):
        #pdb.set_trace()
        self.close_flag = True
        self.server.killServer()