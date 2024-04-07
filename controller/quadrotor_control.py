import numpy as np
from numpy.linalg import norm
import gym
from gym import spaces
import copy

import sys
sys.path.append("/home/fyt/project/quad_raisim")
from controller.quad_utils import *
from controller.sensor_noise import *
import pdb


# jacobian of (acceleration magnitude, angular acceleration)
#       w.r.t (normalized motor thrusts) in range [0, 1]
GRAV = 9.81
EPS = 1e-6
def quadrotor_jacobian(dynamics):
    torque = dynamics.thrust_max * dynamics.prop_crossproducts.T
    torque[2,:] = dynamics.torque_max * dynamics.prop_ccw
    thrust = dynamics.thrust_max * np.ones((1,4))
    dw = (1.0 / dynamics.inertia)[:,None] * torque
    dv = thrust / dynamics.mass
    J = np.vstack([dv, dw])
    J_cond = np.linalg.cond(J)
    # assert J_cond < 100.0
    # if J_cond > 50:
    #     print("WARN: Jacobian conditioning is high: ", J_cond)
    return J

"""
prop_pos = np.array([[0.1591    , -0.1591    ,  0.00846682],
                                        [-0.1591   , -0.1591   ,  0.00846682],
                                        [-0.1591   , 0.1591    ,  0.00846682],
                                        [0.1591    ,  0.1591    ,  0.00846682]]),

                             np.array([[0.11329,  -0.085756 , 0.2166],
                                        [0.11572, -0.40379,  0.2166],
                                        [-0.20529, -0.40597,  0.2166],
                                        [-0.20611,  -0.086572 , 0.2166]]),
"""

class Dynamics():
    def __init__(self, mass=1.5, 
                    thrust_to_weight= 1.939,
                    torque_to_thrust = 0.0104,
                    sense_noise = None,
                    motor_assymetry = np.ones(4),
                    motor_damp_time_up = 0.2,
                    motor_damp_time_down = 0.15,
                    prop_pos = np.array([[0.1591    , -0.1591    ,  0.2166],
                                        [-0.1591   , -0.1591   , 0.2166],
                                        [-0.1591   , 0.1591    ,  0.2166],
                                        [0.1591    ,  0.1591    ,  0.2166]]),
                    prop_ccw = np.array([-1.,  1., -1.,  1.]), #[-1.,  1., -1.,  1.]
                    inertia = np.array([0.00913226, 0.00913226, 0.0175858]),
                    pos = np.zeros(3),
                    vel = np.zeros(3),
                    omega = np.zeros(3),
                    rot = np.eye(3),
                    desired_force=np.zeros(3)):

        self.mass = mass
        self.inertia = inertia
        self.motor_assymetry = motor_assymetry
        self.thrust_to_weight = thrust_to_weight
        self.torque_to_thrust = torque_to_thrust
        self.prop_pos = prop_pos
        
        self.prop_ccw = prop_ccw
        self.motor_damp_time_up = motor_damp_time_up
        self.motor_damp_time_down = motor_damp_time_down
        self.thrust_cmds_damp = np.zeros(4)
        self.thrust_rot_damp = np.zeros(4)
        self.motor_linearity = 1.0
        
        #thrust_noise
        self.thrust_noise_ratio = 0.05
        self.thrust_noise = OUNoise(4, sigma=0.2*self.thrust_noise_ratio)

        #sense_noise
        self.update_sense_noise(sense_noise)

        #pdb.set_trace()
        self.prop_crossproducts = self.get_prop_crossproducts()
        self.thrust_max = self.get_thrust_max()
        self.torque_max = self.get_torque_max()

        self.pos = pos
        self.vel = vel
        self.omega = omega
        self.rot = rot
        self.first_time = copy.deepcopy(pos)
        self.pre_dist = copy.deepcopy(pos)
        self.int_dist = 0.

        #observation_space
        self.omega_max = 40. #rad/s The CF sensor can only show 35 rad/s (2000 deg/s), we allow some extra
        self.vxyz_max = 3.

        #thrust_omega
        self.omega_errls = np.array([0.0,0.0,0.0])

        #contact_force
        self.contact_force = np.array([0., 0., 0.])
        self.contact_torque = np.zeros(3)
        self.desired_force = desired_force
        self.desired_force_world = np.zeros(3)

    def get_prop_crossproducts(self,):
        return np.cross(self.prop_pos, np.array([0., 0., 1.]))

    def get_thrust_max(self,):
        return GRAV * self.mass * self.thrust_to_weight * self.motor_assymetry / 4.0 

    def get_torque_max(self,):
        return self.torque_to_thrust * self.thrust_max
        # change by code_env 20230613 
        #self.torque_max = np.array([0.055562, 0.055562, 0.055562, 0.055562])
        #return self.torque_max

    def update_state(self,pos, vel, omega, rot):
        self.pos = pos
        self.vel = vel
        self.omega = omega
        self.rot = rot

    def angvel2thrust(self, w, linearity=0.424):
        """
        Args:
            linearity (float): linearity factor factor [0 .. 1].
            CrazyFlie: linearity=0.424
        """
        return  (1 - linearity) * w**2 + linearity * w

    def update_sense_noise(self, sense_noise):
        #pdb.set_trace()
        if isinstance(sense_noise, dict):
            self.sense_noise = SensorNoise(**sense_noise)
        elif isinstance(sense_noise, str):
            if sense_noise == "default":
                self.sense_noise = SensorNoise(bypass=False)
            elif sense_noise == "self_define":
                self.sense_noise = SensorNoise(
                    pos_norm_std=0., pos_unif_range=0.,
                    vel_norm_std=0.02,
                    vel_unif_range=0.,
                    quat_norm_std=0.002,
                    quat_unif_range=0.,
                    omega_norm_std=0.06,
                    omega_unif_range=0.,
                    bypass=False,
                    acc_static_noise_std=0,
                    acc_dynamic_noise_ratio=0.005)
            else:
                ValueError("ERROR: QuadEnv: sense_noise parameter is of unknown type: " + str(sense_noise))
        elif sense_noise is None:
            self.sense_noise = SensorNoise(bypass=True)
        else:
            raise ValueError("ERROR: QuadEnv: sense_noise parameter is of unknown type: " + str(sense_noise))

    def step(self,thrust_cmds,dt, flag="body"):
        #print(f"thrust_cmds is {thrust_cmds}")
        thrust_cmds = np.clip(thrust_cmds, a_min=0., a_max=1.)
        self.motor_tau_up = 4*dt/(self.motor_damp_time_up + EPS)
        self.motor_tau_down = 4*dt/(self.motor_damp_time_down + EPS)
        motor_tau = self.motor_tau_up * np.ones([4,])
        motor_tau[thrust_cmds < self.thrust_cmds_damp] = self.motor_tau_down 
        motor_tau[motor_tau > 1.] = 1.

        thrust_rot = thrust_cmds**0.5
        self.thrust_rot_damp = motor_tau * (thrust_rot - self.thrust_rot_damp) + self.thrust_rot_damp       
        self.thrust_cmds_damp = self.thrust_rot_damp**2

        ## Adding noise
        thrust_noise = thrust_cmds * self.thrust_noise.noise()
        self.thrust_cmds_damp = np.clip(self.thrust_cmds_damp + thrust_noise, 0.0, 1.0)        
        #pdb.set_trace()
        thrusts = self.thrust_max * self.angvel2thrust(self.thrust_cmds_damp, linearity=self.motor_linearity)
        
        #trans 4 thrust
        # return 
        
        #Prop crossproduct give torque directions
        torques = self.prop_crossproducts * thrusts[:,None] # (4,3)=(props, xyz)

        # additional torques along z-axis caused by propeller rotations
        #pdb.set_trace()
        torques[:, 2] += self.torque_max * self.prop_ccw * self.thrust_cmds_damp   #四个旋翼的反力矩
        torque = np.sum(torques, axis=0) #np.sum(torques, axis=1)
        #print(f"四个反力矩 {self.torque_max * self.prop_ccw * self.thrust_cmds_damp}")
        thrust = np.sum(thrusts)
        if flag == "prop":
            #pdb.set_trace()
            thrusts = self.thrust_max * thrust_cmds + self.torque_max * self.prop_ccw * self.thrust_cmds_damp 
            #print(f"四个拉力值 {thrusts}")
            return thrusts, torque#np.sum(torques, axis=1)
        elif flag == "body":
            #print(f"四个拉力值 {thrusts}")
            return thrust, torque

    def step2(self,thrust_cmds,dt, flag="body"):
        thrust1, torque1 = self.step(thrust_cmds, dt, flag="body")
        thrust2, torque2 = self.step(thrust_cmds, dt, flag="body")
        return [thrust1,thrust2] , [torque1,torque2]

    def step1(self,thrust_cmds,dt, flag="body"):
        thrust1, torque1 = self.step(thrust_cmds, dt, flag="body")
        return [thrust1] , [torque1]

class NonlinearPositionController(object):
    def __init__(self, dynamics, force_control=False):
        jacobian = quadrotor_jacobian(dynamics)
        self.Jinv = np.linalg.inv(jacobian)
        ## Jacobian inverse for our quadrotor
        # Jinv = np.array([[0.0509684, 0.0043685, -0.0043685, 0.02038736],
        #                 [0.0509684, -0.0043685, -0.0043685, -0.02038736],
        #                 [0.0509684, -0.0043685,  0.0043685,  0.02038736],
        #                 [0.0509684,  0.0043685,  0.0043685, -0.02038736]])
        self.action = None

        #self.kp_p, self.kd_p = 4.5, 3.5
        #self.kp_p, self.kd_p = 4.5, 3.5
        self.kp_p = np.array([4.5, 3.5, 6.4])
        self.kd_p = np.array([4.5, 3.5, 2])
        self.ki_p = np.zeros(3)

        self.kp_a, self.kd_a = 200.0, 70.0 #50.

        #contact_froce 
        self.force_control = force_control
        self.ki_force = 0.1#0.8
        self.e_force = np.zeros(3)

        self.rot_des = np.eye(3)
        # self.rot_des = np.array([[-1.,0.,0.],
        #                         [0.,-1.,0.],
        #                         [0.,0.,1]])

        self.step_func = self.step

        # pidThrustOmega
        self.angle = np.zeros(3)
        self.last_angle = np.zeros(3)
        self.item_omega = np.zeros(3)
        circle_per_sec = 2* np.pi
        self.angle_p_x = 5.0
        self.angle_p_y = 5.0
        self.angle_p_z = 0.0075#0.008#0.015
        self.kpa = np.array([0.000035,8,0.0075]) #  np.array([0.000035,8,0.0075])
        #self.kda = np.array([0.,8,0.]) # np.array([0.,8,0.])
        self.angle_i = np.zeros(3)
        self.e_pi = np.zeros(3)

        # self.kpa = np.array([20,20,20])
        # self.kda = np.array([7.,7,7])

        max_rp =  0.1 * circle_per_sec
        max_yaw =  0.1 * circle_per_sec
        self.min_omega = np.array([ -max_rp, -max_rp, -max_yaw])
        self.max_omega = np.array([  max_rp,  max_rp,  max_yaw])

    def action_space(self, dynamics):
        circle_per_sec = 2 * np.pi
        max_rp =  circle_per_sec # 4 * circle_per_sec
        max_yaw = circle_per_sec # 2 * circle_per_sec
        min_g = -1.0
        max_g = dynamics.thrust_to_weight - 1.0
        self.low  = np.array([min_g, -max_rp, -max_rp, -max_yaw])
        self.high = np.array([max_g,  max_rp,  max_rp,  max_yaw])
        return spaces.Box(self.low, self.high, dtype=np.float32)

    # modifies the dynamics in place.
    def step(self, dynamics, goal, action=None, dt=0.01, flag="body"):
        #print(f"goal is {goal}, pos is {dynamics.pos}")
        to_goal = goal - dynamics.pos
        goal_dist = norm(to_goal)
        e_p = -clamp_norm(to_goal, 4.0)
        e_v = dynamics.vel
        
        #desired force 应该是在机体坐标系下进行描述的 所以需要转换成世界坐标系（需要缕一缕）
        #[Hybrid Force/Motion Control and Internal Dynamics of Quadrotors for Tool Operation] 
        #####################################################################################
        self.e_force += dynamics.contact_force - dynamics.desired_force_world
        force_item = dynamics.desired_force_world - self.ki_force * self.e_force * dt #dynamics.contact_torque
        force_item[0] = 0.
        force_item[2] = 0.
        if not self.force_control or (not np.any(dynamics.contact_force)): #判断是否接触力为0
            force_item[1] = 0.
        # if np.any(dynamics.contact_force):
        #     pdb.set_trace()
        acc_des = -self.kp_p * e_p - self.kd_p * e_v + np.array([0, 0, GRAV]) - force_item/dynamics.mass
        
        
        #acc_des = -self.kp_p * e_p - self.kd_p * e_v + np.array([0, 0, GRAV])

        xc_des = self.rot_des[:, 0] 

        zb_des, _ = normalize(acc_des)
        yb_des, _ = normalize(cross(zb_des, xc_des))
        xb_des    = cross(yb_des, zb_des)
        R_des = np.column_stack((xb_des, yb_des, zb_des))
        R = dynamics.rot
        #print(R)
        def vee(R):
            return np.array([R[2,1], R[0,2], R[1,0]])
        e_R = 0.5 * vee(np.matmul(R_des.T, R) - np.matmul(R.T, R_des))
        e_R[2] *= 0.2 # slow down yaw dynamics
        e_w = dynamics.omega

        dw_des = -self.kp_a * e_R - self.kd_a * e_w
        #这里的dw_des就是期望的角加速度，可以在这里把控制截断做rl的base   20230508

        thrust_mag = np.dot(acc_des, R[:,2])
        #print(f"thrust_mag is {thrust_mag}")
        des = np.append(thrust_mag, dw_des)
        thrusts = np.matmul(self.Jinv, des)
        thrusts[thrusts < 0] = 0
        thrusts[thrusts > 1] = 1
        force, torque = dynamics.step1(thrusts, dt,flag)
        return force, torque

    def stepThrustOmega(self, dynamics, goal, action=None, dt=0.01, flag="body"):
        #print(f"goal is {goal}, pos is {dynamics.pos}")
        #print("stepthrustOmega")
        self.kp_p = np.array([4.5, 3.5, 6.4])
        self.kd_p = np.array([4.5, 3.5, 2])
        #self.ki_p = np.array([0., 0., 2])

        #self.kpa = np.array([2.5,  2.5, 6.])
        #self.kpa = np.array([0.7, 0.7, 0.])
        self.kpa = np.array([2.5,  2.5, 6.])
        self.kda = np.array([1.7, 0.7, 0.])

        self.ki_force = 0.5

        to_goal = goal - dynamics.pos
        goal_dist = norm(to_goal)
        e_p = -clamp_norm(to_goal, 4.0)
        e_v = dynamics.vel
        
        #desired force 应该是在机体坐标系下进行描述的 所以需要转换成世界坐标系（需要缕一缕）
        #[Hybrid Force/Motion Control and Internal Dynamics of Quadrotors for Tool Operation] 
        #####################################################################################
        self.e_force += dynamics.contact_force - dynamics.desired_force_world
        force_item = dynamics.desired_force_world - self.ki_force * self.e_force * dt #dynamics.contact_torque
        force_item[0] = 0.
        force_item[2] = 0.
        if not self.force_control or (not np.any(dynamics.contact_force)): #判断是否接触力为0
            force_item[1] = 0.
        #force_item[1] = 0.
        acc_des = -self.kp_p * e_p - self.kd_p * e_v + np.array([0, 0, GRAV]) - force_item/dynamics.mass
        
        
        #acc_des = -self.kp_p * e_p - self.kd_p * e_v + np.array([0, 0, GRAV])

        xc_des = self.rot_des[:, 0] 

        zb_des, _ = normalize(acc_des)
        yb_des, _ = normalize(cross(zb_des, xc_des))
        xb_des    = cross(yb_des, zb_des)
        R_des = np.column_stack((xb_des, yb_des, zb_des))
        R = dynamics.rot
        self.angle = rotationMatrixToEulerAngles(R)

        # des_omega = (self.angle - self.last_angle)/dt
        #print(R)
        def vee(R):
            return np.array([R[2,1], R[0,2], R[1,0]])
        e_R = vee(np.matmul(R_des.T, R) - np.matmul(R.T, R_des))
        e_R[2] *= 0.2 # slow down yaw dynamics
        e_w = dynamics.omega
        

        #dw_des = -self.kp_a * e_R - self.kd_a * e_w
        
        thrust_mag = np.dot(acc_des, R[:,2])

        item_thrust = thrust_mag / GRAV - 1

        dw_des_2 = -self.kpa * e_R - self.kda * e_w
        action = [item_thrust, dw_des_2[0], dw_des_2[1], dw_des_2[2]]
        new_action = np.array(action)
        return new_action
    
    def stepThrustOmegaForMpc(self, dynamics, goal, action=None, dt=0.01, flag="body"):
        #print(f"goal is {goal}, pos is {dynamics.pos}")
        #print("stepthrustOmega")
        self.kp_p = np.array([4.5, 3.5, 6.4])
        self.kd_p = np.array([4.5, 3.5, 2])
        #self.ki_p = np.array([0., 0., 2])

        # self.kpa = np.array([2.5,  2.5, 6.])
        # self.kda = np.array([1.7, 0.7, 0.])
        self.kpa = np.array([10,  10, 10])
        self.kda = np.array([1.7, 0.7, 0.])

        self.ki_force = 0.5

        to_goal = goal - dynamics.pos
        goal_dist = norm(to_goal)
        e_p = -clamp_norm(to_goal, 4.0)
        e_v = dynamics.vel
        
        #desired force 应该是在机体坐标系下进行描述的 所以需要转换成世界坐标系（需要缕一缕）
        #[Hybrid Force/Motion Control and Internal Dynamics of Quadrotors for Tool Operation] 
        #####################################################################################
        self.e_force += dynamics.contact_force - dynamics.desired_force_world
        force_item = dynamics.desired_force_world - self.ki_force * self.e_force * dt #dynamics.contact_torque
        force_item[0] = 0.
        force_item[2] = 0.
        if not self.force_control or (not np.any(dynamics.contact_force)): #判断是否接触力为0
            force_item[1] = 0.
        #force_item[1] = 0.
        acc_des = -self.kp_p * e_p - self.kd_p * e_v + np.array([0, 0, GRAV]) - force_item/dynamics.mass
        
        
        #acc_des = -self.kp_p * e_p - self.kd_p * e_v + np.array([0, 0, GRAV])

        xc_des = self.rot_des[:, 0] 

        zb_des, _ = normalize(acc_des)
        yb_des, _ = normalize(cross(zb_des, xc_des))
        xb_des    = cross(yb_des, zb_des)
        R_des = np.column_stack((xb_des, yb_des, zb_des))
        R = dynamics.rot
        self.angle = rotationMatrixToEulerAngles(R)

        # des_omega = (self.angle - self.last_angle)/dt
        #print(R)
        def vee(R):
            return np.array([R[2,1], R[0,2], R[1,0]])
        e_R = vee(np.matmul(R_des.T, R) - np.matmul(R.T, R_des))
        e_R[2] *= 0.2 # slow down yaw dynamics
        e_w = dynamics.omega
        

        #dw_des = -self.kp_a * e_R - self.kd_a * e_w
        
        thrust_mag = np.dot(acc_des, R[:,2])

        #item_thrust = thrust_mag / GRAV - 1
        item_thrust = thrust_mag
        dw_des_2 = -self.kpa * e_R - self.kda * e_w
        action = [item_thrust, dw_des_2[0], dw_des_2[1], dw_des_2[2]]
        new_action = np.array(action)
        return new_action
    
    def step_pos_50hz(self, dynamics, goal, action=None, dt=0.01, flag="body"):
        #pdb.set_trace()
        # self.kp_p = np.array([4.5, 3.5, 6.4])
        # self.kd_p = np.array([4.5, 3.5, 2])
        
        self.kp_p = np.array([0.1, 3.5, 18.])
        self.kd_p = np.array([0.2, 3.5, 10.])
        self.ki_p = np.array([0., 0.0, 0.135])
        
        # self.kpa = np.array([2.5,  2.5, 6.])
        # self.kda = np.array([1.7, 0.7, 0.])
        self.kpa = np.array([3.8,  2.5, 6.])
        self.kda = np.array([2., 0.7, 0.])
        
        to_goal = goal - dynamics.pos
        goal_dist = norm(to_goal)
        e_p = -clamp_norm(to_goal, 4.0)
        e_v = dynamics.vel
        
        #desired force 应该是在机体坐标系下进行描述的 所以需要转换成世界坐标系（需要缕一缕）
        #[Hybrid Force/Motion Control and Internal Dynamics of Quadrotors for Tool Operation] 
        #####################################################################################
        self.e_force += dynamics.contact_force - dynamics.desired_force_world
        force_item = dynamics.desired_force_world - self.ki_force * self.e_force * dt #dynamics.contact_torque
        force_item[0] = 0.
        force_item[2] = 0.
        if not self.force_control or (not np.any(dynamics.contact_force)): #判断是否接触力为0
            force_item[1] = 0.
        # if np.any(dynamics.contact_force):
        #     pdb.set_trace()
        acc_des = -self.kp_p * e_p - self.kd_p * e_v + np.array([0, 0, GRAV]) - force_item/dynamics.mass
        
        
        #acc_des = -self.kp_p * e_p - self.kd_p * e_v + np.array([0, 0, GRAV])

        xc_des = self.rot_des[:, 0] 

        zb_des, _ = normalize(acc_des)
        yb_des, _ = normalize(cross(zb_des, xc_des))
        xb_des    = cross(yb_des, zb_des)
        R_des = np.column_stack((xb_des, yb_des, zb_des))
        R = dynamics.rot
        #print(R)
        def vee(R):
            return np.array([R[2,1], R[0,2], R[1,0]])
        e_R = 0.5 * vee(np.matmul(R_des.T, R) - np.matmul(R.T, R_des))
        e_R[2] *= 0.2 # slow down yaw dynamics
        e_w = dynamics.omega

        dw_des = -self.kp_a * e_R - self.kd_a * e_w
        #这里的dw_des就是期望的角加速度，可以在这里把控制截断做rl的base   20230508

        thrust_mag = np.dot(acc_des, R[:,2])
        #print(f"thrust_mag is {thrust_mag}")
        des = np.append(thrust_mag, dw_des)
        thrusts = np.matmul(self.Jinv, des)
        thrusts[thrusts < 0] = 0
        thrusts[thrusts > 1] = 1
        force, torque = dynamics.step1(thrusts, dt,flag)
        return force, torque
    
    def stepThrustOmega50hz(self, dynamics, goal, action=None, dt=0.01, flag="body"):
        
        # self.kp_p = np.array([4.5, 3.5, 6.4])
        # self.kd_p = np.array([4.5, 3.5, 2])
        
        self.kp_p = np.array([0.1, 3.5, 18.])
        self.kd_p = np.array([0.2, 3.5, 10.])
        self.ki_p = np.array([0., 0.0, 0.06])
        
        # self.kpa = np.array([2.5,  2.5, 6.])
        # self.kda = np.array([1.7, 0.7, 0.])
        self.kpa = np.array([3.8,  2.5, 6.])
        self.kda = np.array([2., 0.7, 0.])
        
        
        # self.kp_p = np.array([2.5, 2.5, 0.1])
        # self.kd_p = np.array([5., 2., 5.])
        #self.ki_p = np.array([0., 0., 2])

        # self.kpa = np.array([0.2,  0.05, 0.])
        # self.kda = np.array([6., 6., 2.])
        # self.kpa = np.array([2.62,  2.62, 6.]) #[2.62,2.62,6.0]
        # self.kda = np.array([1.9, 0.816, 0.]) #[1.9,0.815,0.]

        self.ki_force = 0.5

        to_goal = goal - dynamics.pos
        goal_dist = norm(to_goal)
        e_p = -clamp_norm(to_goal, 0.5)
        e_v = dynamics.vel
        self.e_pi += e_p
        
        #desired force 应该是在机体坐标系下进行描述的 所以需要转换成世界坐标系（需要缕一缕）
        #[Hybrid Force/Motion Control and Internal Dynamics of Quadrotors for Tool Operation] 
        #####################################################################################
        self.e_force += dynamics.contact_force - dynamics.desired_force_world
        force_item = dynamics.desired_force_world - self.ki_force * self.e_force * dt #dynamics.contact_torque
        force_item[0] = 0.
        force_item[2] = 0.
        if not self.force_control or (not np.any(dynamics.contact_force)): #判断是否接触力为0
            force_item[1] = 0.
        #force_item[1] = 0.
        #acc_des = -self.kp_p * e_p - self.kd_p * e_v + np.array([0, 0, GRAV]) - force_item/dynamics.mass
        #acc_des = - self.kp_p * e_p - self.kd_p * e_v + np.array([0, 0, GRAV])

        
        acc_des = -self.ki_p * self.e_pi - self.kp_p * e_p - self.kd_p * e_v + np.array([0, 0, GRAV])

        xc_des = self.rot_des[:, 0] 

        zb_des, _ = normalize(acc_des)
        yb_des, _ = normalize(cross(zb_des, xc_des))
        xb_des    = cross(yb_des, zb_des)
        R_des = np.column_stack((xb_des, yb_des, zb_des))
        R = dynamics.rot
        self.angle = rotationMatrixToEulerAngles(R)

        # des_omega = (self.angle - self.last_angle)/dt
        #print(R)
        def vee(R):
            return np.array([R[2,1], R[0,2], R[1,0]])
        e_R = vee(np.matmul(R_des.T, R) - np.matmul(R.T, R_des))
        e_R[2] *= 0.2 # slow down yaw dynamics
        e_w = dynamics.omega
        

        #dw_des = -self.kp_a * e_R - self.kd_a * e_w
        
        thrust_mag = np.dot(acc_des, R[:,2])

        item_thrust = thrust_mag / GRAV - 1

        dw_des_2 = -self.kpa * e_R - self.kda * e_w
        action = [item_thrust, dw_des_2[0], dw_des_2[1], dw_des_2[2]]
        new_action = np.array(action)
        return new_action

class VelocityYawControl(object):
    def __init__(self, dynamics):
        jacobian = quadrotor_jacobian(dynamics)
        self.Jinv = np.linalg.inv(jacobian)
        self.step_func = self.step

    def action_space(self, dynamics):
        vmax = 20.0 # meters / sec
        dymax = 4 * np.pi # radians / sec
        high = np.array([vmax, vmax, vmax, dymax])
        return spaces.Box(-high, high, dtype=np.float32)

    def step(self, dynamics,  goal, action, dt, flag="body",observation=None):
        # needs to be much bigger than in normal controller
        # so the random initial actions in RL create some signal
        #action[3] = 0.

        kp_v = np.array([5., 5., 190]) #193 #195.2
        kp_a, kd_a = 100, 50
        # kp_a = np.array([100.,  100., 100.])
        # kd_a = np.array([50., 50., 50.])

        # kp_a = np.array([2.5,  2.5, 6.])
        # kd_a = np.array([1.7, 0.7, 0.])

        e_v = dynamics.vel - action[:3]
        acc_des = -kp_v * e_v + np.array([0, 0, GRAV])

        # rotation towards the ideal thrust direction
        # see Mellinger and Kumar 2011
        R = dynamics.rot
        R_des = np.eye(3)
        e_R = 0.5 * np.matmul(R_des.T, R) - np.matmul(R.T, R_des)
        
        #期望的偏航角速度
        omega_des = np.array([0, 0, action[3]])
        
        e_w = dynamics.omega - omega_des
        dw_des = -kp_a * e_R - kd_a * e_w
    
        des = np.append(thrust_mag, dw_des)
        thrusts = np.matmul(self.Jinv, des)
        #thrusts = np.clip(thrusts, a_min=0.0, a_max=1.0)
        thrusts[thrusts < 0] = 0
        thrusts[thrusts > 1] = 1
        force, torque = dynamics.step1(thrusts, dt,flag)
        return force, torque


    ##def step(self, dynamics, action, goal, dt, observation=None):
    #def step(self, dynamics, goal, action=None, dt=0.01, flag="body"):
    def step_sou(self, dynamics,  goal, action, dt, flag="body",observation=None):
        # needs to be much bigger than in normal controller
        # so the random initial actions in RL create some signal
        #action[3] = 0.

        kp_v = np.array([5., 5., 190]) #193 #195.2
        kp_a, kd_a = 100, 50
        # kp_a = np.array([100.,  100., 100.])
        # kd_a = np.array([50., 50., 50.])

        # kp_a = np.array([2.5,  2.5, 6.])
        # kd_a = np.array([1.7, 0.7, 0.])

        e_v = dynamics.vel - action[:3]
        acc_des = -kp_v * e_v + np.array([0, 0, GRAV])

        # rotation towards the ideal thrust direction
        # see Mellinger and Kumar 2011
        R = dynamics.rot
        zb_des, _ = normalize(acc_des)
        yb_des, _ = normalize(cross(zb_des, R[:,0]))
        xb_des    = cross(yb_des, zb_des)
        R_des = np.column_stack((xb_des, yb_des, zb_des))
        #R_des = np.eye(3)

        def vee(R):
            return np.array([R[2,1], R[0,2], R[1,0]])
        e_R = 0.5 * vee(np.matmul(R_des.T, R) - np.matmul(R.T, R_des))
        #期望的偏航角速度
        omega_des = np.array([0, 0, action[3]])
        
        e_w = dynamics.omega - omega_des

        dw_des = -kp_a * e_R - kd_a * e_w
        # we want this acceleration, but we can only accelerate in one direction!
        thrust_mag = np.dot(acc_des, dynamics.rot[:,2])

        des = np.append(thrust_mag, dw_des)
        thrusts = np.matmul(self.Jinv, des)
        #thrusts = np.clip(thrusts, a_min=0.0, a_max=1.0)
        thrusts[thrusts < 0] = 0
        thrusts[thrusts > 1] = 1
        force, torque = dynamics.step1(thrusts, dt,flag)
        return force, torque


class OmegaThrustControl(object):
    def __init__(self, dynamics, force_control=False):
        # import pdb
        # pdb.set_trace()
        jacobian = quadrotor_jacobian(dynamics)
        # self.vv = jacobian
        self.Jinv = np.linalg.inv(jacobian)
        self.step_func = self.step
        self.omega_errls = [0,0,0]
        self.omega_errlast = None

        #contact_froce 
        self.force_control = force_control
        self.ki_force = 0.1#0.8
        self.e_force = np.zeros(3)

    def action_space(self, dynamics):
        circle_per_sec = 2 * np.pi
        max_rp =  circle_per_sec # 4 * circle_per_sec
        max_yaw = circle_per_sec # 2 * circle_per_sec
        min_g = -1.0
        max_g = dynamics.thrust_to_weight - 1.0
        self.low  = np.array([min_g, -max_rp, -max_rp, -max_yaw, min_g, -max_rp, -max_rp, -max_yaw])
        self.high = np.array([max_g,  max_rp,  max_rp,  max_yaw, min_g, -max_rp, -max_rp, -max_yaw])
        return spaces.Box(self.low, self.high, dtype=np.float32)

    # def step_self(self, dynamics, action, goal, dt, observation=None):
    #     # action[action < 0] = 0
    #     # action[action > 1] = 1
    #     action = np.clip(action, a_min=self.low, a_max=self.high)
    #     dynamics.step_thrustAngleVel(action, dt)
    #     dynamics.step_thrustAngleVel(action, dt)
    def integrator( self,input,orign,ls,t,x):
        ls[x] = ls[x]+input*t
        output = ls[x] + orign
        return output
    # modifies the dynamics in place.
    #def step(self, dynamics, action, goal, dt, observation=None):
    def step_mpc(self,  dynamics, goal, dt, action=None, flag="body",observation=None):
        # kp = random.randint(9,11) # could be more aggressive
        #pdb.set_trace()
        # pdb.set_trace()
        # kp =  random.uniform(9,13)
        #print("thrustOmega")
        kp = 12.5
        ki = 2
        kd = 0
        kff = 0
        kpp = 10
        kpi = 0
        kpd = 0
        kpff = 0
        krp = 10
        kri = 0
        krd = 0
        krff = 0
        kyp = 10
        kyi = 0
        kyd = 0
        kyff = 0
        detla_time = 0.05 #0.01
        # dynamics.omega = action[1:]
        # action[0] = 0.5
        # action[1] = 2 * np.pi
        # action[2] = 2 * np.pi
        # action[3] = 1 * np.pi
        
        #pdb.set_trace()
        #bc train 20230618
        ######################
        # thrust = action[0]
        # action = action * 0.8
        # action[0] = thrust
        ###########################
        #action[0] -= 0.5

        omega_err = dynamics.omega - action[1:]
        i_factor = np.array([0.,0.,0.])
        i_factor[0] = omega_err[0]/7
        i_factor[1] = omega_err[1]/7
        i_factor[2] = omega_err[2]/7
        d_input = (omega_err - (self.omega_errlast if (self.omega_errlast is not None) else omega_err))/dt
        self.omega_errlast = omega_err 
        # omega_err = np.array([1.0,1.0,1.0])
        # err_omega_integral_x = self.integrator(err_omega[0],0,err_omega_integral_ls,detla_time,0)
        # err_omega_integral_y = self.integrator(err_omega[1],0,err_omega_integral_ls,detla_time,1)
        # err_omega_integral_z = self.integrator(err_omega[2],0,err_omega_integral_ls,detla_time,2)
        dynamics.omega_errls[0] += omega_err[0] * detla_time * (1 - i_factor[0]*i_factor[0])
        dynamics.omega_errls[1] += omega_err[1] * detla_time * (1 - i_factor[1]*i_factor[1])
        dynamics.omega_errls[2] += omega_err[2] * detla_time * (1 - i_factor[2]*i_factor[2])
        dw_des = -kp * omega_err - ki * dynamics.omega_errls + kd * d_input - kff * action[1:]
        # dw_des = -kp * omega_err
        # action[0] = dynamics.thrust_to_weight - 1.0
        acc_des = GRAV * (action[0] + 1.0) 
        
        # rnd = np.random.normal(loc=0.0, scale=0.6,size=1)
        # acc_des = acc_des + rnd
        #acc_des = action[0] / dynamics.mass
        #acc_des = (action[0] +1) / dynamics.mass - GRAV
        des = np.append(acc_des, dw_des)
        thrusts = np.matmul(self.Jinv, des)
        # pdb.set_trace()
        # vvfin = np.matmul(self.vv, thrusts)
        thrusts[thrusts < 0] = 0
        thrusts[thrusts > 1] = 1
        force, torque = dynamics.step1(thrusts, dt, flag)
        return force, torque
    
    def step(self,  dynamics, goal, dt, action=None, flag="body",observation=None):
        # kp = random.randint(9,11) # could be more aggressive
        #pdb.set_trace()
        # pdb.set_trace()
        # kp =  random.uniform(9,13)
        #print("thrustOmega")
        kp = 10
        ki = 2
        kd = 0.0
        kff = 0
        kpp = 10
        kpi = 0
        kpd = 0
        kpff = 0
        krp = 10
        kri = 0
        krd = 0
        krff = 0
        kyp = 10
        kyi = 0
        kyd = 0
        kyff = 0
        detla_time = 0.05 #0.01
        # dynamics.omega = action[1:]
        # action[0] = 0.5
        # action[1] = 2 * np.pi
        # action[2] = 2 * np.pi
        # action[3] = 1 * np.pi
        
        #pdb.set_trace()
        #bc train 20230618
        ######################
        # thrust = action[0]
        # action = action * 0.8
        # action[0] = thrust
        ###########################
        #action[0] -= 0.5

        omega_err = dynamics.omega - action[1:]
        i_factor = np.array([0.,0.,0.])
        i_factor[0] = omega_err[0]/7
        i_factor[1] = omega_err[1]/7
        i_factor[2] = omega_err[2]/7
        d_input = (omega_err - (self.omega_errlast if (self.omega_errlast is not None) else omega_err))/dt
        self.omega_errlast = omega_err 
        # omega_err = np.array([1.0,1.0,1.0])
        # err_omega_integral_x = self.integrator(err_omega[0],0,err_omega_integral_ls,detla_time,0)
        # err_omega_integral_y = self.integrator(err_omega[1],0,err_omega_integral_ls,detla_time,1)
        # err_omega_integral_z = self.integrator(err_omega[2],0,err_omega_integral_ls,detla_time,2)
        dynamics.omega_errls[0] += omega_err[0] * detla_time * (1 - i_factor[0]*i_factor[0])
        dynamics.omega_errls[1] += omega_err[1] * detla_time * (1 - i_factor[1]*i_factor[1])
        dynamics.omega_errls[2] += omega_err[2] * detla_time * (1 - i_factor[2]*i_factor[2])
        dw_des = -kp * omega_err - ki * dynamics.omega_errls + kd * d_input - kff * action[1:]
        # dw_des = -kp * omega_err
        # action[0] = dynamics.thrust_to_weight - 1.0
        acc_des = GRAV * (action[0] + 1.0) 
        
        # rnd = np.random.normal(loc=0.0, scale=0.6,size=1)
        # acc_des = acc_des + rnd
        #acc_des = action[0] / dynamics.mass
        #acc_des = (action[0] +1) / dynamics.mass - GRAV
        des = np.append(acc_des, dw_des)
        thrusts = np.matmul(self.Jinv, des)
        # pdb.set_trace()
        # vvfin = np.matmul(self.vv, thrusts)
        thrusts[thrusts < 0] = 0
        thrusts[thrusts > 1] = 1
        force, torque = dynamics.step1(thrusts, dt, flag)
        return force, torque
    
    def stepForce2(self,  dynamics, goal, dt, action=None, flag="body",observation=None):
        # kp = random.randint(9,11) # could be more aggressive
        kp = 10
        ki = 2
        kd = 0.0
        kff = 0
        kpp = 10
        kpi = 0
        kpd = 0
        kpff = 0
        krp = 10
        kri = 0
        krd = 0
        krff = 0
        kyp = 10
        kyi = 0
        kyd = 0
        kyff = 0
        detla_time = 0.05 #0.01
       
        omega_err = dynamics.omega - action[1:]
        i_factor = np.array([0.,0.,0.])
        i_factor[0] = omega_err[0]/7
        i_factor[1] = omega_err[1]/7
        i_factor[2] = omega_err[2]/7
        d_input = (omega_err - (self.omega_errlast if (self.omega_errlast is not None) else omega_err))/dt
        self.omega_errlast = omega_err 
        
        dynamics.omega_errls[0] += omega_err[0] * detla_time * (1 - i_factor[0]*i_factor[0])
        dynamics.omega_errls[1] += omega_err[1] * detla_time * (1 - i_factor[1]*i_factor[1])
        dynamics.omega_errls[2] += omega_err[2] * detla_time * (1 - i_factor[2]*i_factor[2])
        dw_des = -kp * omega_err - ki * dynamics.omega_errls + kd * d_input - kff * action[1:]
        
        acc_des = GRAV * (action[0] + 1.0)
       
        """
        acc_des需要做力控制处理
        """
        # #desired force 应该是在机体坐标系下进行描述的 所以需要转换成世界坐标系（需要缕一缕）
        # #[Hybrid Force/Motion Control and Internal Dynamics of Quadrotors for Tool Operation] 
        # #####################################################################################
        self.e_force += dynamics.contact_force - dynamics.desired_force_world
        force_item = dynamics.desired_force_world - self.ki_force * self.e_force * dt #dynamics.contact_torque
        force_item[0] = 0.
        force_item[2] = 0.
        if not self.force_control or (not np.any(dynamics.contact_force)): #判断是否接触力为0
            force_item[1] = 0.
        R = dynamics.rot
        #pdb.set_trace()
        #print(f"before deal {acc_des}")
        tmp = acc_des * R[:,2].reshape(-1,1)[:,0] - force_item/dynamics.mass
        acc_des = tmp @ R[:,2]
        #print(f"after deal {acc_des}")
        #acc_des = acc_des - force_item/dynamics.mass
        
        des = np.append(acc_des, dw_des)
        thrusts = np.matmul(self.Jinv, des)
        # pdb.set_trace()
        # vvfin = np.matmul(self.vv, thrusts)
        thrusts[thrusts < 0] = 0
        thrusts[thrusts > 1] = 1
        force, torque = dynamics.step1(thrusts, dt, flag)
        return force, torque

    def stepForce(self,  dynamics, goal, dt, action=None, flag="body",observation=None):
        # kp = random.randint(9,11) # could be more aggressive
        #pdb.set_trace()
        # pdb.set_trace()
        # kp =  random.uniform(9,13)
        kp = 10
        ki = 2
        kd = 0.0
        kff = 0
        kpp = 10
        kpi = 0
        kpd = 0
        kpff = 0
        krp = 10
        kri = 0
        krd = 0
        krff = 0
        kyp = 10
        kyi = 0
        kyd = 0
        kyff = 0
        detla_time = 0.05 #0.01
        # dynamics.omega = action[1:]
        # action[0] = 0.5
        # action[1] = 2 * np.pi
        # action[2] = 2 * np.pi
        # action[3] = 1 * np.pi
        
        #pdb.set_trace()
        #bc train 20230618
        ######################
        # thrust = action[0]
        # action = action * 0.8
        # action[0] = thrust
        ###########################
        #action[0] -= 0.5

        omega_err = dynamics.omega - action[1:]
        i_factor = np.array([0.,0.,0.])
        i_factor[0] = omega_err[0]/7
        i_factor[1] = omega_err[1]/7
        i_factor[2] = omega_err[2]/7
        d_input = (omega_err - (self.omega_errlast if (self.omega_errlast is not None) else omega_err))/dt
        self.omega_errlast = omega_err 
        # omega_err = np.array([1.0,1.0,1.0])
        # err_omega_integral_x = self.integrator(err_omega[0],0,err_omega_integral_ls,detla_time,0)
        # err_omega_integral_y = self.integrator(err_omega[1],0,err_omega_integral_ls,detla_time,1)
        # err_omega_integral_z = self.integrator(err_omega[2],0,err_omega_integral_ls,detla_time,2)
        dynamics.omega_errls[0] += omega_err[0] * detla_time * (1 - i_factor[0]*i_factor[0])
        dynamics.omega_errls[1] += omega_err[1] * detla_time * (1 - i_factor[1]*i_factor[1])
        dynamics.omega_errls[2] += omega_err[2] * detla_time * (1 - i_factor[2]*i_factor[2])
        dw_des = -kp * omega_err - ki * dynamics.omega_errls + kd * d_input - kff * action[1:]
        # dw_des = -kp * omega_err
        # action[0] = dynamics.thrust_to_weight - 1.0
        acc_des = GRAV * (action[0] + 1.0) 
        # rnd = np.random.normal(loc=0.0, scale=0.6,size=1)
        # acc_des = acc_des + rnd
        #acc_des = action[0] / dynamics.mass
        #acc_des = (action[0] +1) / dynamics.mass - GRAV
        des = np.append(acc_des, dw_des)
        thrusts = np.matmul(self.Jinv, des)
        # pdb.set_trace()
        # vvfin = np.matmul(self.vv, thrusts)
        thrusts[thrusts < 0] = 0
        thrusts[thrusts > 1] = 1
        force, torque = dynamics.step1(thrusts, dt, flag)
        return force, torque
    
if __name__ == '__main__':
    dynamics = Dynamics()



