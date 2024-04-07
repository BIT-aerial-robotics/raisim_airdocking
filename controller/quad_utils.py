import numpy as np
import math
from numpy import random as nr
import math
from copy import deepcopy


def rot2D(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s], [s, c]])

def rotZ(theta):
    r = np.eye(4)
    r[:2,:2] = rot2D(theta)
    return r

def randyaw():
    rotz = np.random.uniform(-np.pi, np.pi)
    #rotz = 0
    return rotZ(rotz)[:3,:3]

def to_xyhat(vec):
    v = deepcopy(vec)
    v[2] = 0
    v, _ = normalize(v)
    return v

def eulerAnglesToRotationMatrix(theta) :
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])            
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])   
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])             
    R = np.dot(R_z, np.dot( R_y, R_x))
    #R = np.dot(np.dot(R_z,  R_y), R_x )
    return R

def quatXquat(quat, quat_theta):
    ## quat * quat_theta
    noisy_quat = np.zeros(4)
    noisy_quat[0] = quat[0] * quat_theta[0] - quat[1] * quat_theta[1] - quat[2] * quat_theta[2] - quat[3] * quat_theta[3] 
    noisy_quat[1] = quat[0] * quat_theta[1] + quat[1] * quat_theta[0] - quat[2] * quat_theta[3] + quat[3] * quat_theta[2] 
    noisy_quat[2] = quat[0] * quat_theta[2] + quat[1] * quat_theta[3] + quat[2] * quat_theta[0] - quat[3] * quat_theta[1] 
    noisy_quat[3] = quat[0] * quat_theta[3] - quat[1] * quat_theta[2] + quat[2] * quat_theta[1] + quat[3] * quat_theta[0]
    return noisy_quat

def quat2R(qw, qx, qy, qz):
    R = \
    [[1.0 - 2*qy**2 - 2*qz**2,         2*qx*qy - 2*qz*qw,         2*qx*qz + 2*qy*qw],
     [      2*qx*qy + 2*qz*qw,   1.0 - 2*qx**2 - 2*qz**2,         2*qy*qz - 2*qx*qw],
     [      2*qx*qz - 2*qy*qw,         2*qy*qz + 2*qx*qw,   1.0 - 2*qx**2 - 2*qy**2]]
    return np.array(R)
    
def clamp_norm(x, maxnorm):
    n = np.linalg.norm(x)
    return x if n <= maxnorm else (maxnorm / n) * x

def cross(a, b):
    return np.array([a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]])

def normalize(x):
    n = np.linalg.norm(x)
    if n < 0.00001:
        return x, 0
    return x / n, n

#旋转矩阵转欧拉角
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

def quan2angle(quaternion):
    roll = math.atan2(2 * (quaternion[0] * quaternion[1] + quaternion[2] * quaternion[3]), 1 - 2 * (quaternion[1] * quaternion[1] + quaternion[2] * quaternion[2]))
    pitch = math.asin(2 * (quaternion[0] * quaternion[2] - quaternion[1] * quaternion[3]))
    yaw = math.atan2(2 * (quaternion[0] * quaternion[3] + quaternion[1] * quaternion[2]), 1 - 2 * (quaternion[2] * quaternion[2] + quaternion[3] * quaternion[3]))
    return np.array([roll, pitch, yaw])

# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :
    assert(isRotationMatrix(R))
    
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    
    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])

def quan2rot(quan):
    angle = EulerAndQuaternionTransform(quan)
    rot = eulerAnglesToRotationMatrix(angle)
    return rot

def quan_rot(quan): #quan = [w,x,y,z]
    qw = quan[0]
    qx = quan[1]
    qy = quan[2]
    qz = quan[3]
    return np.array([[ 1 - 2*(qy*qy + qz*qz),   2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy)    ],
                     [ 2*(qx*qy + qw*qz)    ,   1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qw*qx)    ],
                     [ 2*(qx*qz - qw*qy)    ,   2*(qy*qz + qw*qx),     1 - 2*(qx*qx + qy*qy)]])

def rot_quan(rot):
    m11 = rot[0][0]
    m22 = rot[1][1]
    m33 = rot[2][2]
    qw = np.sqrt(m11+m22+m33+1)/2
    qx = (rot[2][1] - rot[1][2]) /(4*qw)
    qy = (rot[0][2] - rot[2][0]) / (4*qw)
    qz = (rot[1][0] - rot[0][1]) / (4*qw)
    return np.array([qw, qx, qy, qz])   
     
def rot2quan(rot):
    angle = rotationMatrixToEulerAngles(rot)
    quan = EulerAndQuaternionTransform(angle)
    return quan

def EulerAndQuaternionTransform( intput_data):
    """
        四元素与欧拉角互换
    """
    data_len = len(intput_data)
    angle_is_not_rad = False
 
    if data_len == 3:
        r = 0
        p = 0
        y = 0
        if angle_is_not_rad: # 180 ->pi
            r = math.radians(intput_data[0]) 
            p = math.radians(intput_data[1])
            y = math.radians(intput_data[2])
        else:
            r = intput_data[0] 
            p = intput_data[1]
            y = intput_data[2]
 
        sinp = math.sin(p/2)
        siny = math.sin(y/2)
        sinr = math.sin(r/2)
 
        cosp = math.cos(p/2)
        cosy = math.cos(y/2)
        cosr = math.cos(r/2)
 
        w = cosr*cosp*cosy + sinr*sinp*siny
        x = sinr*cosp*cosy - cosr*sinp*siny
        y = cosr*sinp*cosy + sinr*cosp*siny
        z = cosr*cosp*siny - sinr*sinp*cosy
        return [w,x,y,z]
 
    elif data_len == 4:
 
        w = intput_data[0] 
        x = intput_data[1]
        y = intput_data[2]
        z = intput_data[3]
 
        r = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        p = math.asin(2 * (w * y - z * x))
        y = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
 
        if angle_is_not_rad : # pi -> 180
            r = math.degrees(r)
            p = math.degrees(p)
            y = math.degrees(y)
        return [r,p,y]

def rand_uniform_rot3d():
    randunit = lambda: normalize(np.random.normal(size=(3,)))[0]
    up = randunit()
    fwd = randunit()
    while np.dot(fwd, up) > 0.95:
        fwd = randunit()
    left, _ = normalize(cross(up, fwd))
    # import pdb; pdb.set_trace()
    up = cross(fwd, left)
    rot = np.column_stack([fwd, left, up])
    return rot


def randomRot(angle = np.pi/10): #eulerAnglesToRotationMatrix获取到的是弧度值
    #roll,pitch = np.random.randint(-60, 60, (2))
    #theta = np.random.randint([0, 0, 0],[1, 1, np.pi], (3))
    theta = np.random.uniform(-angle, angle, size=(3,))
    #theta = np.zeros(3)
    theta[2] = np.pi/20
    theta[0] = 0.
    theta[1] = 0.
    #print(theta*180/np.pi)
    return eulerAnglesToRotationMatrix(theta)

def randomRot(angle = np.pi/3, yaw = np.pi/3): #eulerAnglesToRotationMatrix获取到的是弧度值
    #roll,pitch = np.random.randint(-60, 60, (2))
    #theta = np.random.randint([0, 0, 0],[1, 1, np.pi], (3))
    theta = np.random.uniform(-angle, angle, size=(3,))
    theta[2] = np.random.uniform(-yaw,yaw)
    #theta = np.zeros(3)
    #theta[2] = 0.1
    # theta[0] = 0.
    # theta[1] = 0.
    #print(theta*180/np.pi)
    return eulerAnglesToRotationMatrix(theta)

class OUNoise:
    """Ornstein–Uhlenbeck process"""
    def __init__(self, action_dimension, mu=0, theta=0.15, sigma=0.3):
        """
        @param: mu: mean of noise
        @param: theta: stabilization coeff (i.e. noise return to mean)
        @param: sigma: noise scale coeff
        """
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * nr.randn(len(x))
        self.state = x + dx
        return self.state


if __name__ == "__main__":
    r = np.array([[ 0.72585871,  0.68784383,  0.        ],
        [-0.68784383,  0.72585871,  0.        ],
        [ 0.        ,  0.        ,  1.        ]])
    quan = EulerAndQuaternionTransform(np.array([0, 0, -285/180]))
    print(f"quan is {quan}")
    #print(rotationMatrixToEulerAngles(r) * 180/3.14)