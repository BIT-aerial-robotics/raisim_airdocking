import numpy as np
import pdb
import sys
sys.path.append("/home/fyt/raisim_workspace/raisim_quadrotor/raisim_bit/quad_raisim")
from controller.quad_utils import *
import transforms3d as t3d

EPS = 1e-6

def compute_reward_weighted_sou(dynamics, goal, action, dt, crashed, rew_coeff, action_prev):
    ##################################################
    ## log to create a sharp peak at the goal
    import pdb
    # pdb.set_trace()
    # if(dynamics.vvstart == 0):
    #     dynamics.pre_dist = goal - dynamics.pre_dist
    #     dynamics.vvstart +=1
    first_err = goal - dynamics.first_time
    shaping_dist =  (goal - dynamics.pos) -  (dynamics.pre_dist if (dynamics.pre_dist is not None) else first_err)
    loss_shaping_dist = np.linalg.norm(shaping_dist)
    dynamics.pre_dist = goal - dynamics.pos
    
    # print(f"rew_coeff from compute_reward is : {rew_coeff}")
    dist = np.linalg.norm(goal - dynamics.pos)
    # print(dist)
    dynamics.int_dist = dynamics.int_dist + dist
    loss_int_dist = 0.01 * rew_coeff["pos"] * dynamics.int_dist
    loss_pos = rew_coeff["pos"] * (rew_coeff["pos_log_weight"] * np.log(dist +  EPS ) + rew_coeff["pos_linear_weight"] * dist)
    # loss_posz = np.linalg.norm(goal[2] - dynamics.pos[2])
    # loss_pos = dist

    # dynamics_pos = dynamics.pos
    # print('dynamics.pos', dynamics.pos)

    ##################################################
    ## penalize altitude above this threshold
    # max_alt = 6.0
    # loss_alt = np.exp(2*(dynamics.pos[2] - max_alt))

    ##################################################
    # penalize amount of control effort
    loss_effort = rew_coeff["effort"] * np.linalg.norm(action)
    dact = action - action_prev
    loss_act_change = rew_coeff["action_change"] * (dact[0]**2 + dact[1]**2 + dact[2]**2 + dact[3]**2)**0.5
    # loss_act_change = rew_coeff["action_change"] * (dact[0]**2 + dact[1]**2 + dact[2]**2 )**0.5
    # loss_act_change = rew_coeff["action_change"] * (dact[0]**2  )**0.5
    loss_multiact1 = action[0] - action[1]
    loss_multiact2 = action[0] - action[2]
    loss_multiact3 = action[1] - action[2]
    ##################################################
    ## loss omega  
    loss_omega = rew_coeff["omega"] * np.linalg.norm(dynamics.omega)
    circle_per_sec = 2 * np.pi
    max_rp = 4 * circle_per_sec
    max_yaw = 1 * circle_per_sec
    eomega = np.sum(abs(dynamics.omega))/(9 * circle_per_sec)
    loss_omeganew = -np.clip(eomega,0,1)

    ##################################################
    ## loss velocity
    # dx = goal - dynamics.pos
    # dx = dx / (np.linalg.norm(dx) + EPS)
    
    ## normalized    
    # vel_direct = dynamics.vel / (np.linalg.norm(dynamics.vel) + EPS)
    # vel_magn = np.clip(np.linalg.norm(dynamics.vel),-1, 1)
    # vel_clipped = vel_magn * vel_direct 
    # vel_proj = np.dot(dx, vel_clipped)
    # loss_vel_proj = - rew_coeff["vel_proj"] * dist * vel_proj

    # loss_vel_proj = 0. 
    loss_vel = rew_coeff["vel"] * np.linalg.norm(dynamics.vel)
    # pdb.set_trace()
    # print(loss_vel)
    loss_velz = rew_coeff["vel"] * dynamics.vel[2]
    angle = rotationMatrixToEulerAngles(dynamics.rot) * (180/np.pi)
    loss_angle = rew_coeff["alpha_r"] * np.linalg.norm(angle)
    # loss_yaw = rew_coeff["yaw"] *  np.abs(angle[2])
    ##################################################
    ## Loss orientation
    loss_orient = -rew_coeff["orient"] * dynamics.rot[2,2]
    rotn = t3d.euler.mat2euler(dynamics.rot)
    loss_rot = 0.1 * (180*rotn[0]/np.pi) + 0.1 *(180*rotn[1]/np.pi)  + 0.1 * (180*rotn[2]/np.pi) 
    loss_yaw = -rew_coeff["yaw"] * dynamics.rot[0,0]
    loss_new = -rew_coeff["orient"] * dynamics.rot[1,1]
    # Projection of the z-body axis to z-world axis
    # Negative, because the larger the projection the smaller the loss (i.e. the higher the reward)
    rot_cos = ((dynamics.rot[0,0] +  dynamics.rot[1,1] +  dynamics.rot[2,2]) - 1.)/2.
    #We have to clip since rotation matrix falls out of orthogonalization from time to time
    loss_rotation = rew_coeff["rot"] * np.arccos(np.clip(rot_cos, -1.,1.)) #angle = arccos((trR-1)/2) See: [6]
    loss_attitude = rew_coeff["attitude"] * np.arccos(np.clip(dynamics.rot[2,2], -1.,1.))

    ##################################################
    ## Loss for constant uncontrolled rotation around vertical axis
    loss_spin_z  = rew_coeff["spin_z"]  * abs(dynamics.omega[2])
    # loss_spin_xy = rew_coeff["spin_xy"] * np.linalg.norm(dynamics.omega[:2])
    # loss_spin = rew_coeff["spin"] * np.linalg.norm(dynamics.omega) 
    loss_spin = rew_coeff["spin"] * (dynamics.omega[0]**2 + dynamics.omega[1]**2 + dynamics.omega[2]**2)**0.5 

    ##################################################
    ## loss crash
    loss_crash = rew_coeff["crash"] * float(crashed)

    # reward = -dt * np.sum([loss_pos, loss_effort, loss_alt, loss_vel_proj, loss_crash])
    # rew_info = {'rew_crash': -loss_crash, 'rew_altitude': -loss_alt, 'rew_action': -loss_effort, 'rew_pos': -loss_pos, 'rew_vel_proj': -loss_vel_proj}

    reward = -dt * np.sum([
        loss_pos, 
        loss_effort, 
        loss_crash,
        #loss_rot,
        loss_orient,
        #loss_angle,
        loss_yaw,
        loss_rotation,
        loss_new,
        loss_attitude,
        loss_spin,
        loss_spin_z,
        # loss_spin_xy,
        loss_act_change,
        loss_vel,
        # loss_multiact1,
        # loss_multiact2,
        # loss_multiact3,
        # loss_shaping_dist,
        # loss_velz,
        # loss_int_dist,
        loss_omega
        # loss_omeganew
        # loss_posz
        ])
    
    # reward = 0
    

    rew_info = {
    "rew_main": reward,
    'rew_pos': -loss_pos, 
    'rew_action': -loss_effort, 
    'rew_crash': -loss_crash, 
    "rew_orient": -loss_angle,
    "rew_yaw": -loss_yaw,
    "rew_rot": -loss_rotation,
    "rew_attitude": -loss_attitude,
    "rew_spin": -loss_spin,
    "rew_spin_z": -loss_spin_z,
    # "rew_spin_xy": -loss_spin_xy,
    # "rew_act_change": -loss_act_change,
    "action_prev":action_prev[0],
    "rew_vel": -loss_vel,
    "loss_omega": loss_omega
    }
    # pdb.set_trace()
    # print('reward: ', reward, ' pos:', dynamics.pos, ' action', action)
    # print('pos', dynamics.pos)
    if np.isnan(reward) or not np.isfinite(reward):
        for key, value in locals().items():
            print('%s: %s \n' % (key, str(value)))
        raise ValueError('QuadEnv: reward is Nan')
    return reward, rew_info

def compute_reward_force_track(dynamics, goal, action, dt, crashed, rew_coeff, action_prev):
    #position error
    dist = np.linalg.norm(goal - dynamics.pos)
    loss_pos = rew_coeff["pos"] * (rew_coeff["pos_log_weight"] * np.log(dist +  EPS ) + rew_coeff["pos_linear_weight"] * dist)
    
    #force_error
    
    force_error = np.linalg.norm(dynamics.contact_force - dynamics.desired_force_world)
    loss_force = rew_coeff["force"] * force_error

    #attitude_error
    loss_omega = rew_coeff["omega"] * np.linalg.norm(dynamics.omega)
    loss_spin_z  = rew_coeff["spin_z"]  * abs(dynamics.omega[2])
    loss_spin = rew_coeff["spin"] * (dynamics.omega[0]**2 + dynamics.omega[1]**2 + dynamics.omega[2]**2)**0.5 
    loss_crash = rew_coeff["crash"] * int(crashed) 
    reward = -dt * np.sum([
        loss_pos, 
        loss_force, 
        loss_omega,
        loss_spin_z,
        loss_spin,
        loss_crash
        ])

    rew_info = {
    "rew_main": reward,
    'rew_pos': -loss_pos, 
    'rew_force': -loss_force, 
    "rew_spin": -loss_spin,
    "rew_spin_z": -loss_spin_z,
    "rew_omega": -loss_omega,
    "rew_crash":-loss_crash
    }
    
    if np.isnan(reward) or not np.isfinite(reward):
        for key, value in locals().items():
            print('%s: %s \n' % (key, str(value)))
        raise ValueError('QuadEnv: reward is Nan')
    return reward, rew_info


def compute_reward_weighted_force_track(dynamics, goal, action, dt, crashed, rew_coeff, action_prev):
    ##################################################
    first_err = goal - dynamics.first_time
    shaping_dist =  (goal - dynamics.pos) -  (dynamics.pre_dist if (dynamics.pre_dist is not None) else first_err)
    loss_shaping_dist = np.linalg.norm(shaping_dist)
    dynamics.pre_dist = goal - dynamics.pos
    
    #force_error
    if not rew_coeff["force_flag"]:
        rew_coeff["force"] = 0
        
    force_error = np.linalg.norm(dynamics.contact_force - dynamics.desired_force_world)
    loss_force = rew_coeff["force"] * force_error


    #print(f"rew_coeff from compute_reward is : {rew_coeff}")
    dist = np.linalg.norm(goal - dynamics.pos)
    dynamics.int_dist = dynamics.int_dist + dist
    loss_int_dist = 0.01 * rew_coeff["pos"] * dynamics.int_dist
    
    loss_pos = rew_coeff["pos"] * (rew_coeff["pos_log_weight"] * np.log(dist +  EPS ) + rew_coeff["pos_linear_weight"] * dist)
    #pdb.set_trace()
    ##################################################
    # penalize amount of control effort
    loss_effort = rew_coeff["effort"] * np.linalg.norm(action)
    dact = action - action_prev
    loss_act_change = rew_coeff["action_change"] * (dact[0]**2 + dact[1]**2 + dact[2]**2 + dact[3]**2)**0.5
    
    ##################################################
    ## loss omega  
    loss_omega = rew_coeff["omega"] * np.linalg.norm(dynamics.omega)
    # circle_per_sec = 2 * np.pi
    # max_rp = 4 * circle_per_sec
    # max_yaw = 1 * circle_per_sec
    # eomega = np.sum(abs(dynamics.omega))/(9 * circle_per_sec)
    # loss_omeganew = -np.clip(eomega,0,1)

    ##################################################
    ## loss velocity
    loss_vel = rew_coeff["vel"] * np.linalg.norm(dynamics.vel)
    loss_velz = rew_coeff["vel"] * dynamics.vel[2]
    angle = rotationMatrixToEulerAngles(dynamics.rot) * (180/np.pi)
    loss_angle = rew_coeff["alpha_r"] * np.linalg.norm(angle)

    ##################################################
    ## Loss orientation
    loss_orient = -rew_coeff["orient"] * dynamics.rot[2,2]
    rotn = t3d.euler.mat2euler(dynamics.rot)
    loss_rot = 0.1 * (180*rotn[0]/np.pi) + 0.1 *(180*rotn[1]/np.pi)  + 0.1 * (180*rotn[2]/np.pi) 
    loss_yaw = -rew_coeff["yaw"] * dynamics.rot[0,0]
    loss_new = -rew_coeff["orient"] * dynamics.rot[1,1]
    # Projection of the z-body axis to z-world axis
    # Negative, because the larger the projection the smaller the loss (i.e. the higher the reward)
    rot_cos = ((dynamics.rot[0,0] +  dynamics.rot[1,1] +  dynamics.rot[2,2]) - 1.)/2.
    #We have to clip since rotation matrix falls out of orthogonalization from time to time
    loss_rotation = rew_coeff["rot"] * np.arccos(np.clip(rot_cos, -1.,1.)) #angle = arccos((trR-1)/2) See: [6]
    loss_attitude = rew_coeff["attitude"] * np.arccos(np.clip(dynamics.rot[2,2], -1.,1.))

    ##################################################
    ## Loss for constant uncontrolled rotation around vertical axis
    loss_spin_z  = rew_coeff["spin_z"]  * abs(dynamics.omega[2])
    loss_spin = rew_coeff["spin"] * (dynamics.omega[0]**2 + dynamics.omega[1]**2 + dynamics.omega[2]**2)**0.5 

    ##################################################
    ## loss crash
    loss_crash = rew_coeff["crash"] * float(crashed)

    reward = -dt * np.sum([
        loss_pos, 
        loss_effort, 
        loss_crash,
        loss_orient,
        loss_yaw,
        loss_rotation,
        loss_new,
        loss_attitude,
        loss_spin,
        loss_spin_z,
        loss_act_change,
        loss_vel,
        loss_omega,
        loss_force
        ])

    rew_info = {
    "rew_main": reward,
    'rew_pos': -loss_pos, 
    'rew_effort': -loss_effort, 
    'rew_crash': -loss_crash, 
    "rew_orient": -loss_angle,
    "rew_yaw": -loss_yaw,
    "rew_rot": -loss_rotation,
    "rew_new": -loss_new,
    "rew_attitude": -loss_attitude,
    "rew_spin": -loss_spin,
    "rew_spin_z": -loss_spin_z,
    "rew_action_change": -loss_act_change,
    "action_prev":action_prev[0],
    "rew_vel": -loss_vel,
    "loss_omega": loss_omega,
    "loss_force": -loss_force
    }
    if np.isnan(reward) or not np.isfinite(reward):
        for key, value in locals().items():
            print('%s: %s \n' % (key, str(value)))
        raise ValueError('QuadEnv: reward is Nan')
    return reward, rew_info