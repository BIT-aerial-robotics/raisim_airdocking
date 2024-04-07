import numpy as np
import random
mass = 1.5
GRAV = 9.81
# inertia = np.array([0.00913226, 0.00913226, 0.0175858])
# inertia = np.array([0.0043811, 0.0049501,  0.0075123 ])
#inertia = np.array([0.02479455,0.083429,0.10985194])
inertia = np.array([0.0685339,0.08342368,0.1501083])
#inertia = np.array([0.39990040143520006, 0.11735786261519998, 0.47971748234000006])
#inertia = np.array([0.0067211875,0.0080406875,0.014278875])
#inertia = np.array([0.01840,0.01840,0.020377])
#inertia = 
change_goal_flag = True
control_flag = "body"
control_name = 'pidThrustOmega' # thrustOmegaPolicy #"NonPos" # nonLine, thrustOmega
# goal = np.array([0., 0., 1.5])
# goal2 = np.array([0., -0.9, 1.5]) #np.array([0.,-0.9,1.5])
#stable goal
goal = np.array([0., 1., 1.5])
goal2 = np.array([0., -0.9, 1.5])

#random goal


# gc = [0., 0., 1.0, 0.8775825618903728, 0.0, 0.0, 0.,0.,0.,0.,0.]
# gc2 = [ 0., -11.9, 1.0, 0.8775825618903728, 0.0, 0.0, 0.,0.,0.,0.,0.]

gc = [0., 0., 0.0, 0.8775825618903728, 0.0, 0.0, 0.,0.,0.,0.,0.]
gc2 = [ 0., -11.9, 0.0, 0.8775825618903728, 0.0, 0.0, 0.,0.,0.,0.,0.]
sense_noise = "self_define"#None
#sense_noise = "default"  #None ,self_define

# force  control
force_control = True #False#False
force_y = np.random.uniform(0,3)
desired_force = np.array([0., 1, 0.]) #应该是机体坐标系下的量

interval = 500
#action = np.array([3,3,3,3,3,3, 0.707, 0.707, 0.707, 0.707, 0.707, 0.707])
action = np.array([14,0,0,0])

#space set
room_box = np.array([[-5., -5., 0.2], [5., 5., 6.]])
obs_repr = "xyz_vxyz_rot_omega_force_torque_xyz_vxyz_rot_omega"


#reward
rew_coeff_sou=  {
            "pos": 1.02, "pos_offset": 0., "pos_log_weight": 0., "pos_linear_weight": 0.,
            "effort": 0., 
            "crash": 0., 
            "orient": 0., "yaw": 0., "rot": 0., "attitude": 0.,
            "spin_z": 0., "spin_xy": 0.,
            "spin": 0.,
            "vel": 0.,
            "omega": 0.,
            "action_change":0.,
            "alpha_a":0.,
            "alpha_v":0.,
            "alpha_r":0.,
            "action_mean":0.,
            "thrust_change":0.,
            "thrust_mean":0.,
            "one_p":0.,
            "force":0.,
            "force_flag": False,  #标志是否需要做力跟踪
            "done":0,
            "goal_y":0,
            "goal_xz":0,
            "airdocking":0,
            "crash_rew":0,
            "ka":0.,
            "depth_tau": 0.14,
            "maximun_overlap_reward":2.0,
            "minimun_overlap_reward":0.2,
            "minimun_joint_depth":0.05,
            "overlap_coeff":1.0,
            "position_coeff":1.02,
            "orientation_coeff":0.1

            }